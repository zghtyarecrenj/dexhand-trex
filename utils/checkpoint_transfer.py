"""Audited transfer from the official bimanual policy to a native right-only policy."""
import torch

RESET_PREFIXES = ('x_embedder.', 'final_layer.', 'final_layer_tactile.', 'state_embedder.')
OPTIONAL_PREFIXES = ('flare_', 'tactile_vqvae.', 'tactile_code_embedder.', 'tacf6_vqvae_',
                     'deform_encoder.', 'deform_proj.')


def transfer_state(model, source, source_args, *, audit_only=False):
    """Reuse compatible experts/encoders; never slice robot action or state heads.

    The model must be freshly initialized. Only the pretrained VQ normalization
    buffers have an explicit bimanual-to-right mapping. Dataset normalization
    always comes from the current training split, outside this function.
    """
    if source_args.get('action_dim') != 62 or model.action_dim != 31:
        raise ValueError('Official transfer requires a 62D source and 31D target')
    if source_args.get('action_chunk') != model.action_chunk:
        raise ValueError('Action chunk differs from the official checkpoint')
    target = model.state_dict()
    selected, reset, sliced, skipped = {}, [], [], []
    for name, value in target.items():
        if name.startswith(RESET_PREFIXES):
            reset.append(name)
            continue
        if name not in source:
            raise ValueError(f'Missing shared checkpoint tensor: {name}')
        tensor = source[name]
        if name.startswith('tacf6_vqvae_'):
            if tensor.shape != (60,) or value.shape != (30,):
                raise ValueError(f'Unexpected pretrained VQ normalization shape: {name}')
            tensor = tensor[30:60]
            sliced.append(name)
        if tensor.shape != value.shape:
            raise ValueError(f'Incompatible shared tensor {name}: {tuple(tensor.shape)} -> {tuple(value.shape)}')
        selected[name] = tensor
    for name in source:
        if name not in target:
            if not name.startswith(RESET_PREFIXES + OPTIONAL_PREFIXES):
                raise ValueError(f'Unexpected checkpoint tensor: {name}')
            skipped.append(name)
    report = dict(source_action_dim=62, target_action_dim=31, action_chunk=model.action_chunk,
                  copied_tensors=len(selected), copied_elements=sum(t.numel() for t in selected.values()),
                  target_elements=sum(t.numel() for t in target.values()),
                  reset_tensors=reset, right_vq_stat_slices=sliced, omitted_source_tensors=skipped,
                  action_heads_sliced=False, source_dataset_normalization_loaded=False,
                  audited_only=audit_only)
    if not audit_only:
        missing, unexpected = model.load_state_dict(selected, strict=False)
        if set(missing) != set(reset) or unexpected:
            raise RuntimeError(f'Unexpected transfer result: missing={missing}, unexpected={unexpected}')
    return report
