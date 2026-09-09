"""A complete loss forward and FSDP helpers for the custom cascaded T-Rex model."""
import torch
import torch.nn.functional as F
from torch import nn
from qwen_vla import extend_position_ids_for_flare, split_slow_fast_embeds
from qwen_vla.modeling_qwen3vl_mot import Qwen3VLDecoderLayerMoT
from utils.flow_loss import masked_flow_mse


class PolicyLossModule(nn.Module):
    """Keep every custom policy call inside the distributed root forward."""
    def __init__(self, policy, args):
        super().__init__()
        self.policy = policy
        self.args = args

    def forward(self, batch, validation=False):
        args = self.args
        is_stage1 = args.training_stage == 1
        T_per_frame = args.n_flare_tokens_per_frame
        S_steps = args.n_flare_steps
        K = T_per_frame * S_steps
        use_flare = bool(args.use_flare and K > 0)
        flare_layer_idx = args.flare_layer_index
        raw_model = self.policy

        inputs_embeds = raw_model.prepare_inputs_embeds(
            input_ids=batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
        )

        n_slow_imgs = batch["n_slow_images"]
        grid_thw = batch.get("image_grid_thw")
        B = inputs_embeds.shape[0]
        if grid_thw is not None and grid_thw.shape[0] > n_slow_imgs:
            merge = getattr(raw_model.visual, "spatial_merge_size",
                            2)
            n_imgs_per_sample = grid_thw.shape[0] // B
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in grid_thw[:n_imgs_per_sample][:n_slow_imgs]
            )
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, batch["input_ids"],
                raw_model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds
            fast_embeds = inputs_embeds[:, :0]

        L_slow = slow_embeds.shape[1]

        pos_ids, _ = raw_model.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch["attention_mask"],
        )
        pos_ids = pos_ids[:, :, :L_slow]

        if use_flare:
            flare_q = raw_model.flare_queries.expand(B, -1, -1).to(device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds_ext = torch.cat([slow_embeds, flare_q], dim=1)
            pos_ids = extend_position_ids_for_flare(pos_ids, K)
            L_latent = slow_embeds_ext.shape[1]  # L_slow + K
        else:
            slow_embeds_ext = slow_embeds
            L_latent = L_slow

        if args.use_robot_state and batch["state_raw"] is not None:
            state_vec = batch["state_raw"].to(slow_embeds.device,
                                               dtype=slow_embeds.dtype)
            state_embeds = raw_model.state_embedder(state_vec).unsqueeze(1)
        else:
            state_embeds = torch.empty((B, 0, slow_embeds.shape[2]), device=slow_embeds.device, dtype=slow_embeds.dtype)
        n_state = state_embeds.shape[1]

        noisy_actions = raw_model.x_embedder(
            batch["noisy_actions"].to(slow_embeds.dtype))
        timesteps = raw_model.t_embedder(
            batch["timesteps"].to(slow_embeds.dtype)).unsqueeze(1)

        chunk = args.action_chunk
        target = batch["target"].to(slow_embeds.dtype)
        n_fast = fast_embeds.shape[1]
        r_target_norm = None  # set only when tactile expert ran this step

        has_any_tac = bool(args.use_tactile_vec or args.use_tactile_deform or args.use_tactile_code or args.use_tactile_vqvae)

        # Cascaded: shared flow split between action + tactile experts.
        # 1) L_flow — action expert single-step (tactile-blind), trained on
        #    full τ ∈ [0, 1] so the action expert can run standalone for
        #    graceful degradation when tactile is missing.
        full_embeds = torch.cat([
            slow_embeds_ext,
            fast_embeds, state_embeds, timesteps, noisy_actions,
        ], dim=1)
        L_total = full_embeds.shape[1]
        outputs = raw_model.model(
            inputs_embeds=full_embeds,
            position_ids=pos_ids,
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=use_flare,
            latent_indexes=torch.arange(0, L_latent, device=full_embeds.device),
            action_indexes=torch.arange(L_latent, L_total, device=full_embeds.device),
            tactile_indexes=torch.arange(0, 0, device=full_embeds.device),
        )
        hidden = outputs.last_hidden_state
        act_pred_start = L_latent + n_fast + n_state + 1
        v_act = raw_model.final_layer(
            hidden[:, act_pred_start: act_pred_start + chunk, :])
        loss_act = masked_flow_mse(v_act, target, batch.get("action_is_pad"))

        if has_any_tac and not is_stage1:
            fe = fast_embeds if n_fast > 0 else None
            se = state_embeds if n_state > 0 else None
            # cached_kv must summarize the action expert's state at
            # τ=τ_split — exactly what the tactile expert attends to at
            # inference.  forward_flow_action_partial caches τ=τ_split
            # keys (refresh_clean_kv=True writes the partially-denoised
            # x_split into the KV).
            ahat_noise = torch.randn_like(batch["noisy_actions"])
            with torch.no_grad():
                _, cached_kv, n_action_in_cache, _ = (
                    raw_model.forward_flow_action_partial(
                        inputs_embeds=slow_embeds_ext,
                        position_ids=pos_ids,
                        noise=ahat_noise,
                        attention_mask=batch["attention_mask"],
                        state_embeds=se,
                        fast_embeds=fe,
                        num_steps_total=args.cascaded_total_steps,
                        split_step=args.cascaded_split_step,
                        refresh_clean_kv=True,
                    ))

            # 2) L_flow_tactile — tactile expert predicts velocity at
            #    τ ∈ [0, τ_split] for the SAME flow distribution as the
            #    action expert (no residual).  Same target ε − A_demo.
            norm_actions_gt = batch["norm_actions"].to(slow_embeds.dtype)
            eps_r    = batch["eps_r"].to(slow_embeds.dtype)
            tau_split = 1.0 - args.cascaded_split_step / args.cascaded_total_steps
            # Reuse existing time_r ∈ (0, 1] (Beta-sampled) and rescale to (0, τ_split].
            tau_t    = batch["time_r"].to(slow_embeds.dtype) * tau_split
            tau_t_b  = tau_t[:, None, None]
            x_tau    = (1 - tau_t_b) * norm_actions_gt + tau_t_b * eps_r
            v_target_r = eps_r - norm_actions_gt
            # Diagnostic: magnitude of the velocity target the tactile
            # expert is being asked to predict.
            r_target_norm = v_target_r.float().abs().mean().detach()

            # Tactile dropout — zero the tactile signal (not the tensor
            # shape) so the tactile expert learns "all-zero tactile" as
            # a graceful-degradation fallback.  Nulling the tensors
            # would break _embed_tactile_observations, which requires
            # at least one of {codes, f6, deform} to be present.
            drop = (not validation and args.cascaded_tactile_dropout > 0
                    and torch.rand((), device=full_embeds.device).item()
                    < args.cascaded_tactile_dropout)
            tac_f6_in    = batch.get("tactile_f6s_delayed")
            tac_def_in   = batch.get("tactile_deforms_delayed")
            tac_codes_in = batch.get("tactile_codes")
            tac_hist_in  = batch.get("tactile_f6_history")
            if drop:
                if tac_f6_in is not None:
                    tac_f6_in = torch.zeros_like(tac_f6_in)
                if tac_def_in is not None:
                    tac_def_in = torch.zeros_like(tac_def_in)
                if tac_codes_in is not None:
                    tac_codes_in = torch.zeros_like(tac_codes_in)
                if tac_hist_in is not None:
                    # Zero raw F6 → a consistent "no-contact" code on encode.
                    tac_hist_in = torch.zeros_like(tac_hist_in)

            loss_tac = masked_flow_mse(
                raw_model.tactile_flow_train_step(
                    cached_kv=cached_kv,
                    latent_position_ids=pos_ids,
                    n_action_in_cache=n_action_in_cache,
                    x_tau=x_tau,
                    tau=tau_t,
                    tactile_f6=tac_f6_in,
                    tactile_deform=tac_def_in,
                    tactile_codes=tac_codes_in,
                    tactile_f6_history=tac_hist_in,
                ),
                v_target_r, batch.get("action_is_pad"),
            )
        else:
            loss_tac = 0.0

        loss_flare = 0.0
        if not validation and use_flare and batch["flare_pixel_values"] is not None:
            # Extract flare hidden states from intermediate layer or last layer
            if flare_layer_idx == -1:
                flare_source = hidden  # last layer (already normed)
            else:
                # outputs.hidden_states is a tuple: (embed, layer0, layer1, ..., layerN)
                # flare_layer_idx can be negative (e.g. -7 for ~3/4 depth)
                all_hs = outputs.hidden_states
                # Skip index 0 (embedding), so layer indices are 1..N
                n_layers = len(all_hs) - 1  # exclude embedding
                if flare_layer_idx < 0:
                    layer_i = n_layers + flare_layer_idx  # e.g. 28 + (-7) = 21
                else:
                    layer_i = flare_layer_idx
                layer_i = max(0, min(layer_i, n_layers - 1))
                flare_source = all_hs[layer_i + 1]  # +1 to skip embedding

            flare_hidden = flare_source[:, L_slow: L_slow + K, :]  # [B, K, H]
            flare_pred = raw_model.flare_proj(flare_hidden)  # [B, K, H]

            f_pv = batch["flare_pixel_values"].to(device=flare_pred.device, dtype=flare_pred.dtype)
            f_thw = batch["flare_grid_thw"].to(device=flare_pred.device)

            with torch.no_grad():
                vit_out = raw_model.visual(f_pv, grid_thw=f_thw)
                features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

                # Adaptive pool each frame to T_per_frame tokens
                merge = getattr(raw_model.visual, "spatial_merge_size", 2)
                frame_feats = []
                offset = 0
                for g in f_thw:
                    n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
                    frame_tokens = features[offset: offset + n_tok]  # [n_tok, H]
                    # Adaptive average pool: [n_tok, H] → [T_per_frame, H]
                    pooled = F.adaptive_avg_pool1d(
                        frame_tokens.unsqueeze(0).permute(0, 2, 1),  # [1, H, n_tok]
                        T_per_frame,
                    ).permute(0, 2, 1).squeeze(0)  # [T_per_frame, H]
                    frame_feats.append(pooled)
                    offset += n_tok
                # frame_feats: list of B*S_steps tensors, each [T_per_frame, H]
                # Reshape to [B, S_steps * T_per_frame, H] = [B, K, H]
                flare_targets = torch.stack(frame_feats).view(B, K, -1)

            # Cosine similarity loss
            pred_norm = F.normalize(flare_pred, dim=-1)
            tgt_norm = F.normalize(flare_targets.detach(), dim=-1)
            loss_flare = (1.0 - (pred_norm * tgt_norm).sum(dim=-1)).mean()

        loss = loss_act
        if torch.is_tensor(loss_tac):
            loss = loss + args.cascaded_loss_weight * loss_tac
        if torch.is_tensor(loss_flare):
            loss = loss + args.flare_loss_weight * loss_flare


        return {"loss": loss, "action_loss": loss_act,
                "tactile_loss": loss_tac if torch.is_tensor(loss_tac) else loss.new_zeros(()),
                "flare_loss": loss_flare if torch.is_tensor(loss_flare) else loss.new_zeros(()),
                "r_target_norm": r_target_norm}


def wrap_fsdp(model, device):
    """FULL_SHARD each decoder block and the root; frozen sensor towers stay replicated."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    if device.type == "cpu":
        device = torch.device("cpu")  # Accelerate may report cpu:0; FSDP compares devices exactly.
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise ValueError("FSDP requires torchrun with at least two processes")
    ignored = [getattr(model.policy, name, None) for name in ("visual", "deform_encoder", "tactile_vqvae")]
    ignored = [m for m in ignored if m is not None and not any(p.requires_grad for p in m.parameters())]
    for module in ignored:
        module.to(device)
    return FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=ModuleWrapPolicy({Qwen3VLDecoderLayerMoT}),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=None),
                ignored_modules=ignored, use_orig_params=True, limit_all_gathers=True,
                backward_prefetch=None, sync_module_states=device.type == "cuda")


def policy_state_dict(model, accelerator):
    """All ranks must call this collective; return portable, unprefixed policy weights."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    if isinstance(model, FSDP):
        config = FullStateDictConfig(offload_to_cpu=accelerator.device.type == "cuda", rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            state = model.state_dict()
    else:
        state = accelerator.get_state_dict(model)
    return {k.removeprefix("policy."): v for k, v in state.items()}


def clip_grad_norm(model, accelerator, max_norm):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        return model.clip_grad_norm_(max_norm)
    return accelerator.clip_grad_norm_(model.parameters(), max_norm)
