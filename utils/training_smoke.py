"""Small actual T-Rex models for offline gradient checks; no pretrained weights."""
import torch
from transformers import Qwen3VLTextConfig
from qwen_vla import Qwen3VLVLAModel
from tactile_vqvae.models.tactile_vqvae import TactileVQVAEConfig
from utils.flow_loss import masked_flow_mse


def small_model(dim=31, vqvae=False, granularity='hand', deform=False):
    config = Qwen3VLTextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
        rope_scaling={'rope_type': 'default', 'mrope_section': [2, 3, 3]})
    vqconfig = TactileVQVAEConfig(hidden_channels=16, bottleneck_channels=16,
        embed_dim=8, codebook_size=8, granularity=granularity).to_dict()
    model = Qwen3VLVLAModel(config, action_dim=dim, action_chunk=16,
        use_robot_state=True, use_tactile_deform=deform, use_tactile_vqvae=vqvae,
        vqvae_config=vqconfig if vqvae else None, vqvae_codebook_size=8)
    model.initialize_vla_weights()
    return model


def check_cascaded_backward(actions, f6, padding, deforms=None, supervision=None):
    """Actual MoT, both velocity heads, cache flow and optimizer; synthetic vision tokens."""
    torch.manual_seed(7)
    model = small_model(deform=deforms is not None).to(torch.bfloat16)
    if deforms is not None:
        model.deform_encoder.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch, chunk, width = actions.shape
    latent = torch.randn(batch, 4, 64, dtype=torch.bfloat16)
    wrist = torch.randn(batch, 2, 64, dtype=torch.bfloat16)
    positions = torch.arange(4)[None,None,:].expand(3,batch,-1)
    state = model.state_embedder(actions[:,0].to(torch.bfloat16)).unsqueeze(1)
    time = torch.full((batch,), .6, dtype=torch.bfloat16)
    noise = torch.randn_like(actions).to(torch.bfloat16)
    actions = actions.to(torch.bfloat16)
    noisy = .6 * noise + .4 * actions
    embeds = torch.cat([latent, wrist, state, model.t_embedder(time).unsqueeze(1),
                        model.x_embedder(noisy)], dim=1)
    out = model.model(inputs_embeds=embeds, position_ids=positions, use_cache=False,
        latent_indexes=torch.arange(4), action_indexes=torch.arange(4,embeds.shape[1]),
        tactile_indexes=torch.arange(0))
    action_prediction = model.final_layer(out.last_hidden_state[:,-chunk:])
    action_prediction.retain_grad()
    loss_action = masked_flow_mse(action_prediction, noise-actions, padding, supervision)
    with torch.no_grad():
        _, cache, count, _ = model.forward_flow_action_partial(latent, positions, noise,
            fast_embeds=wrist, state_embeds=state, num_steps_total=2, split_step=1)
    tactile_prediction = model.tactile_flow_train_step(cache, positions, count,
        .2 * noise + .8 * actions, torch.full((batch,), .2),
        tactile_f6=f6, tactile_deform=deforms)
    tactile_prediction.retain_grad()
    loss_tactile = masked_flow_mse(tactile_prediction, noise-actions, padding, supervision)
    (loss_action + loss_tactile).backward()
    for predicted in (action_prediction, tactile_prediction):
        assert predicted.shape == (batch,chunk,width)
        assert torch.count_nonzero(predicted.grad[padding]) == 0
        assert torch.count_nonzero(predicted.grad[~padding]) > 0
        if supervision is not None:
            assert torch.count_nonzero(predicted.grad[~supervision]) == 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    for head in (model.final_layer, model.final_layer_tactile, model.tacf6_embedder):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
    before = model.final_layer.mlp.fc2.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.final_layer.mlp.fc2.weight)
    return {'action_loss':loss_action.item(), 'tactile_loss':loss_tactile.item(),
            'output_shape':list(action_prediction.shape), 'finite_gradients':True,
            'padding_gradient_zero':True, 'optimizer_step_changed_weights':True,
            'model':'actual T-Rex MoT, one layer, hidden size 64, random weights',
            'vision':'synthetic embeddings; pretrained visual tower not tested',
            'deform_encoder':bool(deforms is not None)}

