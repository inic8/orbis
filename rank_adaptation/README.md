# Orbis Rank Adaptation

This directory contains the repo-local SVD rank-adaptation workflow for Orbis. It is self-contained and does not depend on the deleted `dlr/` package.

The workflow targets `nn.Linear` layers inside the Orbis transformer backbone, runs a rank sweep, allocates a per-layer compression budget, applies sequential low-rank replacement, and writes a new Orbis run directory with the adapted checkpoint and reports.

Rank-adapted checkpoints also embed low-rank reconstruction metadata so they can be reloaded later by rank adaptation and pruning without losing the altered module structure.

The rank-adaptation output is designed to be chainable into the local pruning workflow. A verified integration run from the original Orbis checkpoint to rank adaptation and then pruning succeeded in this repository.

## Main entry points

CLI:

```bash
python -m rank_adaptation.cli --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt
```

Python API:

```python
from rank_adaptation import OrbisRankAdaptationOptions, rank_adapt_orbis_checkpoint

result = rank_adapt_orbis_checkpoint(
    checkpoint_path="logs_wm/orbis_288x512/checkpoints/last.ckpt",
    options=OrbisRankAdaptationOptions(
        acc_budget_pct=2.0,
        comp_target=2.0,
        rank_step_fraction=0.20,
        min_features=64,
        run_benchmark=True,
    ),
)
```

## What it does

- Loads an Orbis checkpoint and config using the local bootstrap logic.
- Finds compressible `nn.Linear` layers under `model.vit` by name pattern.
- Runs Phase 1 rank sweeps unless cached checkpoint JSON files are provided.
- Fits a noise-to-quality polynomial per layer.
- Distributes the global compression budget across layers.
- Applies cumulative low-rank replacements and keeps accepted candidates.
- Writes a new Orbis-style run directory under `logs_wm/` by default.

## Default output layout

If `--output-dir` is omitted, outputs are written to:

```text
logs_wm/<source_run>_rank_adapted/
```

That run directory contains:

- `config.yaml`
- `checkpoints/last.ckpt`
- `onnx/`
- `rank_adaptation/rank_adaptation_stats.json`
- `rank_adaptation/rank_adaptation_summary.json`
- `rank_adaptation/phase1_checkpoints/*.json` when Phase 1 is run
- `rank_adaptation/benchmark_stats.json` when benchmarking is enabled

The saved checkpoint includes:

- `state_dict`
- `rank_adaptation_stats`
- `rank_adaptation_metadata`

## Basic usage

Minimal command:

```bash
python -m rank_adaptation.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt
```

Explicit config and output directory:

```bash
python -m rank_adaptation.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --config logs_wm/orbis_288x512/config.yaml \
  --output-dir logs_wm/orbis_288x512_rank_adapted
```

Reuse cached Phase 1 sweep results:

```bash
python -m rank_adaptation.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --checkpoint-dir logs_wm/orbis_288x512_rank_adapted/rank_adaptation/phase1_checkpoints \
  --skip-phase1
```

More aggressive compression example:

```bash
python -m rank_adaptation.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --acc-budget-pct 3.0 \
  --comp-target 2.5 \
  --rank-step-fraction 0.10 \
  --min-features 128
```

Run rank adaptation first and then prune the resulting checkpoint:

```bash
python -m rank_adaptation.cli \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --output-dir logs_wm/orbis_288x512_rank_adapted

python -m pruning.cli \
  --checkpoint logs_wm/orbis_288x512_rank_adapted/checkpoints/last.ckpt \
  --config logs_wm/orbis_288x512_rank_adapted/config.yaml \
  --output-dir logs_wm/orbis_288x512_rank_adapted_pruned
```

Run rank adaptation through the shared optimization pipeline:

```bash
python optimization_pipeline.py \
  --checkpoint logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --config logs_wm/orbis_288x512/config.yaml \
  --steps rank_adaptation
```

To chain rank adaptation with pruning in either order and emit a single `pipeline_summary.json`, see [optimization_pipeline.py](../optimization_pipeline.py) and the top-level [README.md](../readme.md).

## CLI options

- `--checkpoint`: source checkpoint path.
- `--output-dir`: optional output run directory override.
- `--config`: optional config override. If omitted, `config.yaml` is detected near the checkpoint.
- `--orbis-repo`: optional path to an Orbis checkout or parent repo root.
- `--acc-budget-pct`: global quality budget percentage.
- `--comp-target`: target compression ratio for compressible linear layers.
- `--rank-step-fraction`: phase-1 sweep step size as a fraction of `min(in_features, out_features)`.
- `--min-features`: ignore very small linear layers.
- `--batch-size`: saved with the options payload for bookkeeping.
- `--vit-attr`: transformer backbone attribute name, default `vit`.
- `--checkpoint-dir`: directory containing cached Phase 1 JSON sweep artifacts.
- `--skip-phase1`: skip generating fresh phase-1 sweep results.
- `--skip-benchmark`: disable synthetic before/after latency and memory benchmarking.

## Result object

`rank_adapt_orbis_checkpoint()` returns an `OrbisRankAdaptationResult` with:

- `output_dir`
- `checkpoint_path`
- `config_path`
- `stats_path`
- `summary_path`
- `benchmark_path`
- `stats`
- `output_artifacts`
- `component_result`

## Notes

- The workflow expects an Orbis checkpoint that loads into a model exposing `model.vit` by default.
- If the checkpoint was produced in the standard repo layout, config detection usually works without extra arguments.
- Benchmarking uses synthetic inputs and is meant for relative before/after comparison, not production latency certification.
- Checkpoints saved by this workflow preserve low-rank reconstruction metadata so downstream pruning can rebuild the adapted module structure before loading weights.