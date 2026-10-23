"""Explicit source splits, exact distributed replay, and clean paired validation.

Training loss/augmentation are intentionally unchanged. Samplers shard once;
do not pass these loaders through Accelerate's second sharding layer.
"""
import contextlib
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from accelerate.utils import send_to_device


class MixedBatchSampler(Sampler):
    """Exact 3:1 quota per rank; shuffled cycles without row duplication on disk.

    All ranks construct identical global streams and select disjoint slices.
    Each pool is permuted without replacement before starting another cycle.
    An epoch is explicitly `steps` global batches, not a pass over the union.
    """
    def __init__(self, original, dagger, batch_size, rank, world_size, steps, seed=42):
        if not original or not dagger or batch_size % 4 or batch_size < 4:
            raise ValueError('Both pools and a local batch divisible by four are required')
        if not 0 <= rank < world_size or steps < 1:
            raise ValueError('Invalid rank/world size/epoch length')
        if set(original) & set(dagger):
            raise ValueError('Dataset pools overlap')
        self.pools = [list(original), list(dagger)]
        self.quota = [batch_size * 3 // 4, batch_size // 4]
        self.rank, self.world_size, self.steps, self.seed = rank, world_size, steps, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        streams = []
        for pool, quota in zip(self.pools, self.quota):
            needed = self.steps * self.world_size * quota
            cycles = [rng.permutation(pool) for _ in range(math.ceil(needed / len(pool)))]
            streams.append(np.concatenate(cycles)[:needed].reshape(self.steps, self.world_size, quota))
        for step in range(self.steps):
            batch = np.concatenate([s[step, self.rank] for s in streams])
            # Same local permutation is harmless; membership differs by rank.
            yield rng.permutation(batch).tolist()


class ExactValidationSampler(Sampler):
    """Equal collective counts, no repeated metric rows; -1 is masked padding."""
    def __init__(self, size, batch_size, rank, world_size):
        if size < 1 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid validation partition')
        self.size, self.batch_size, self.rank, self.world_size = size, batch_size, rank, world_size

    def __len__(self):
        return math.ceil(self.size / (self.batch_size * self.world_size))

    def __iter__(self):
        for step in range(len(self)):
            start = (step * self.world_size + self.rank) * self.batch_size
            yield [i if i < self.size else -1 for i in range(start, start + self.batch_size)]


class RankLoader(DataLoader):
    def set_epoch(self, epoch):
        self.batch_sampler.set_epoch(epoch)


@contextlib.contextmanager
def fixed_randomness(seed, cuda=False):
    """Paired flow samples for validation, restoring the training RNG stream."""
    py, np_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if cuda and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed); np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py); np.random.set_state(np_state)


def validation_seed(rows):
    identity = '|'.join(str(r.get('mixed_row_id', i)) for i, r in enumerate(rows))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')


def validate_split_rows(rows):
    assignments = {}
    for row in rows:
        if row['dataset_id'] not in ('original', 'dagger') or row['split'] not in ('train', 'val'):
            raise ValueError('Missing dataset identity or explicit split')
        source = row['source_episode_id']
        if not str(source).startswith(row['dataset_id'] + ':'):
            raise ValueError('Source identity must be namespaced')
        old = assignments.setdefault(source, row['split'])
        if old != row['split']:
            raise ValueError('Source appears in both training and validation')


def build_loaders(dataset, accelerator, args):
    if args.gradient_accumulation_steps != 1 or args.distributed_backend != 'fsdp':
        raise ValueError('Mixed pilot requires native FSDP and accumulation=1')
    rows = dataset.hf_dataset
    # Inspect only identity columns, avoiding deserialization of action/media arrays.
    identities = rows.select_columns(['dataset_id', 'split', 'source_episode_id'])
    validate_split_rows(identities)
    val_loaders = {}
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    for kind in ('original', 'dagger'):
        indices = [i for i, r in enumerate(identities) if r['split']=='val' and r['dataset_id']==kind]
        val = copy.copy(dataset)
        val.hf_dataset = rows.select(indices)
        val.training = False
        sampler = ExactValidationSampler(len(val), args.train_bsz_per_gpu,
                                        accelerator.process_index, accelerator.num_processes)
        val_loaders[kind] = DataLoader(val, batch_sampler=sampler, collate_fn=val.collate_fn,
                                      generator=torch.Generator().manual_seed(args.validation_seed), **kw)
    train_indices = [i for i, r in enumerate(identities) if r['split']=='train']
    dataset.hf_dataset = rows.select(train_indices)
    kinds = dataset.hf_dataset['dataset_id']
    pools = [[i for i, kind in enumerate(kinds) if kind == k] for k in ('original', 'dagger')]
    sampler = MixedBatchSampler(*pools, args.train_bsz_per_gpu, accelerator.process_index,
                                accelerator.num_processes, args.mixed_steps_per_epoch, args.seed)
    loader = RankLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn,
                        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index), **kw)
    return loader, val_loaders


def valid_counts(batch):
    valid = (~batch['action_is_pad']).unsqueeze(-1) & batch['action_supervision_mask']
    real = batch.get('metric_real_rows', torch.ones(valid.shape[0], device=valid.device))
    return valid.sum().double(), real.sum().double()


def accumulate_validation(totals, outputs, batch):
    count, rows = valid_counts(batch)
    totals[0] += outputs['action_loss'].detach().double() * count
    totals[1] += outputs['tactile_loss'].detach().double() * count
    totals[2] += count
    totals[3] += rows


@torch.no_grad()
def run_validation(model, loaders, accelerator, args):
    was_training = model.training
    model.eval()
    metrics = {}
    try:
        for kind, loader in loaders.items():
            totals = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
            loader.generator.manual_seed(args.validation_seed)
            for i, batch in enumerate(loader):
                if args.max_val_batches > 0 and i >= args.max_val_batches:
                    break
                batch = send_to_device(batch, accelerator.device)
                seed = int(batch['validation_rng_seed']) + args.validation_seed
                with fixed_randomness(seed, cuda=True), accelerator.autocast():
                    out = model(batch, validation=True)
                if not torch.isfinite(out['loss']):
                    raise RuntimeError(f'Nonfinite {kind} validation loss')
                accumulate_validation(totals, out, batch)
            if dist.is_initialized():
                dist.all_reduce(totals)
            if totals[2] <= 0:
                raise ValueError(f'No valid targets in {kind} validation')
            if not args.max_val_batches and int(totals[3]) != len(loader.dataset):
                raise RuntimeError('Validation rows were repeated or omitted')
            metrics.update({f'val/{kind}/action_loss': (totals[0]/totals[2]).item(),
                            f'val/{kind}/tactile_loss': (totals[1]/totals[2]).item(),
                            f'val/{kind}/valid_scalars': int(totals[2]),
                            f'val/{kind}/rows': int(totals[3])})
    finally:
        model.train(was_training)
    return metrics


def write_metrics(args, accelerator, step, kind, values):
    if accelerator.is_main_process:
        with (Path(args.output_dir)/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step, kind=kind, **values))+'\n')


def sampling_counts(batch, accelerator, total):
    ids = batch['dataset_id']
    counts = torch.stack([(ids==i).sum() for i in (0, 1)])
    if dist.is_initialized():
        gathered = [torch.zeros_like(counts) for _ in range(accelerator.num_processes)]
        dist.all_gather(gathered, counts)
    else:
        gathered = [counts]
    expected = torch.tensor([len(ids)*3//4, len(ids)//4], device=counts.device)
    if any(not torch.equal(c, expected) for c in gathered):
        raise RuntimeError('Observed per-rank mixing quota differs from 75/25')
    total += torch.stack(gathered).sum(0)
    return {'sampled/original': int(total[0]), 'sampled/dagger': int(total[1]),
            'sampled/original_fraction': (total[0]/total.sum()).item(),
            'sampled/per_rank_this_update': [c.tolist() for c in gathered]}
