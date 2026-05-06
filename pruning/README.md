# Orbis Pruning

This directory contains the local Orbis structured pruning workflow. It prunes MLP blocks inside the world-model transformer and writes a new repo-local run directory with the pruned checkpoint, config, and reports.

The pruning path also supports checkpoints that were first processed by [rank_adaptation/README.md](../rank_adaptation/README.md). If the input checkpoint contains low-rank reconstruction metadata, pruning rebuilds that module structure before loading weights.

## Main entry points

CLI:

```bash
python -m pruning.cli --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt
```

Python API:

```python
from pruning import OrbisPruningOptions, prune_orbis_checkpoint

result = prune_orbis_checkpoint(
    checkpoint_path="logs_wm/orbis_288x512/checkpoints/last.ckpt",
    options=OrbisPruningOptions(
        mlp_prune_ratio=0.2,
        mlp_round_to=128,
        head_prune_ratio=0.0,
        mlp_prune_layers="all",
        importance_metric="l1_weight",
    ),
)
```

## What it does

- Loads an Orbis checkpoint and config from the local repo layout.
- Reconstructs low-rank modules first if the checkpoint came from rank adaptation.
- Applies structured MLP pruning to `Mlp` and `SwiGLU` modules in `model.vit`.
- Updates the saved config for the new MLP ratio when pruning changes the hidden dimension.
- Writes a new Orbis-style run directory under `logs_wm/` by default.

## Default output layout

If `--output-dir` is omitted, outputs are written to:

```text
logs_wm/<source_run>_pruned/
```

That run directory contains:

- `config.yaml`
- `checkpoints/last.ckpt`
- `onnx/`
- `pruning/pruning_stats.json`
- `pruning/model_structure_summary.json`
- `pruning/benchmark_stats.json` when benchmarking is enabled

## Basic usage

Minimal command:

```bash
python -m pruning.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt
```

Prune a rank-adapted checkpoint:

```bash
python -m pruning.cli \
  --checkpoint logs_wm/orbis_288x512_rank_adapted/checkpoints/last.ckpt \
  --config logs_wm/orbis_288x512_rank_adapted/config.yaml \
  --output-dir logs_wm/orbis_288x512_rank_adapted_pruned
```

Example with lighter MLP pruning:

```bash
python -m pruning.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --mlp-prune-ratio 0.10 \
  --mlp-round-to 64 \
  --mlp-prune-layers space
```

Run pruning through the shared optimization pipeline:

```bash
python optimization_pipeline.py \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --config logs_wm/orbis_288x512/config.yaml \
  --steps pruning
```

To chain pruning with rank adaptation in either order and emit a single `pipeline_summary.json`, see [optimization_pipeline.py](../optimization_pipeline.py) and the top-level [README.md](../readme.md).

## CLI options

- `--checkpoint`: source checkpoint path.
- `--output-dir`: optional output run directory override.
- `--config`: optional config override. If omitted, `config.yaml` is detected near the checkpoint.
- `--orbis-repo`: optional path to an Orbis checkout or parent repo root.
- `--mlp-prune-ratio`: fraction of MLP hidden width to remove.
- `--mlp-round-to`: round the pruned MLP width to this multiple.
- `--head-prune-ratio`: attention head pruning request. The local fallback pruner keeps this at `0.0`.
- `--mlp-prune-layers`: prune `all`, `space`, or `time` MLP layers.
- `--head-prune-layers`: retained for interface symmetry.
- `--importance-metric`: one of `l1_weight`, `l2_weight`, or `random`.
- `--skip-benchmark`: disable synthetic before/after latency and memory benchmarking.

## Notes

- The local fallback implementation currently targets MLP pruning only.
- Pruning can follow rank adaptation because the loader reconstructs saved low-rank layers before loading the checkpoint state dict.
- Benchmarking uses synthetic inputs and is intended for relative comparisons only.