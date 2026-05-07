# ONNX Export Workflow

This directory contains the ONNX export path for `model.vit`, optional deployment-target profiling for Orin and Thor, and a set of debug helpers used to validate intermediate submodules before trusting the final exported graph.

The main entry point is:

```bash
python export/export_to_device.py --run-dir logs_wm/orbis_288x512/
```

## What gets exported

The exporter wraps `model.vit` and writes a single ONNX file for the world-model transformer path. It does not export the full training stack or Lightning module.

Optionally, the same workflow can also build a TensorRT engine from the exported ONNX artifact. The TensorRT stage is kept generic by using deployment-target profiles and explicit input-shape limits rather than hard-coding a single model variant.

By default, the script resolves artifacts from the run directory:

- `config.yaml`
- `checkpoints/*.ckpt` using the most recent checkpoint by modification time
- `onnx/last.onnx` if no ONNX file exists yet, otherwise the most recent `onnx/*.onnx`

If TensorRT building is enabled, the default engine artifact is written under:

- `tensorrt/<onnx_stem>_<deployment_target>_<precision>.engine`

For the default repo layout this means a command like:

```bash
python export/export_to_device.py --run-dir logs_wm/orbis_288x512/
```

typically uses:

- `logs_wm/orbis_288x512/config.yaml`
- `logs_wm/orbis_288x512/checkpoints/last.ckpt`
- `logs_wm/orbis_288x512/onnx/last.onnx`

You can override any of these paths explicitly:

```bash
python export/export_to_device.py \
  --config-path logs_wm/orbis_288x512/config.yaml \
  --ckpt-path logs_wm/orbis_288x512/checkpoints/last.ckpt \
  --onnx-path logs_wm/orbis_288x512/onnx/custom_name.onnx
```

## Basic usage

Minimal command:

```bash
python export/export_to_device.py --run-dir logs_wm/orbis_288x512/
```

Useful options:

```bash
python export/export_to_device.py \
  --run-dir logs_wm/orbis_288x512/ \
  --batch-size 1 \
  --context-frames 4 \
  --target-frames 1 \
  --num-samples 3 \
  --opset 17 \
  --atol 1e-4 \
  --rtol 1e-4
```

Build an Orin-oriented TensorRT engine after ONNX export:

```bash
python export/export_to_device.py \
  --run-dir logs_wm/orbis_288x512/ \
  --deployment-target orin \
  --build-tensorrt
```

Build a Thor-oriented TensorRT engine with explicit profile limits:

```bash
python export/export_to_device.py \
  --run-dir logs_wm/orbis_288x512/ \
  --deployment-target thor \
  --build-tensorrt \
  --trt-max-batch-size 8 \
  --trt-max-context-frames 8 \
  --trt-max-target-frames 2
```

If `--build-tensorrt` is requested and the chosen ONNX file already exists, the workflow reuses that ONNX artifact and skips the ONNX preflight, export, and parity stages.

Available knobs:

- `--run-dir`: experiment directory containing `config.yaml`, `checkpoints/`, and usually `onnx/`
- `--config-path`: explicit config override
- `--ckpt-path`: explicit checkpoint override
- `--onnx-path`: explicit output path override
- `--block-index`: STDiT block used for block-level parity checks
- `--swin-module-index`: which Swin module instance to validate when multiple exist
- `--window-attention-index`: which `WindowAttention` module instance to validate
- `--num-samples`: number of random parity samples per check in addition to the captured sample
- `--batch-size`: dummy/export batch size
- `--context-frames`: number of context frames in dummy inputs
- `--target-frames`: number of target frames in dummy inputs
- `--opset`: ONNX opset version, default `17`
- `--atol`: absolute tolerance for parity checks
- `--rtol`: relative tolerance for parity checks
- `--disable-constant-folding`: export without constant folding
- `--deployment-target`: one of `generic`, `orin`, or `thor`; selects default TensorRT profile settings
- `--build-tensorrt`: also build a TensorRT engine from the exported ONNX artifact
- `--engine-path`: explicit TensorRT engine output path override
- `--tensorrt-backend`: TensorRT build backend, one of `auto`, `python`, or `trtexec`
- `--trt-workspace-gb`: TensorRT workspace memory pool size in GB
- `--trt-opt-batch-size`: TensorRT optimization profile batch size
- `--trt-max-batch-size`: TensorRT optimization profile maximum batch size
- `--trt-max-context-frames`: maximum context-frame count in the TensorRT optimization profile
- `--trt-max-target-frames`: maximum target-frame count in the TensorRT optimization profile
- `--trt-fp16` / `--trt-no-fp16`: force-enable or disable FP16 engine building

## Deployment targets

The export workflow now exposes three deployment-target profiles:

- `generic`: neutral defaults suitable for general TensorRT deployment
- `orin`: lower default batch ceilings for Jetson Orin-style deployment
- `thor`: larger default workspace and batch ceilings for Thor-class deployment

These profiles only affect TensorRT defaults. ONNX export and ONNX parity checks remain the same across targets.

If you need tighter deployment constraints, override the profile defaults explicitly with the `--trt-*` shape and workspace flags.

## Expected workflow output

The script is intentionally methodical. A successful run is split into four stages for ONNX-only export, or five stages when TensorRT building is requested.

### 1. Loading model

The script prints the resolved paths and the comparison tolerance, then loads the model from config and checkpoint.

Example header:

```text
========================================================================
Methodical ONNX Export Workflow
========================================================================
Config      /mnt/models/orbis/logs_wm/orbis_288x512/config.yaml
Checkpoint /mnt/models/orbis/logs_wm/orbis_288x512/checkpoints/last.ckpt
ONNX       /mnt/models/orbis/logs_wm/orbis_288x512/onnx/last.onnx
Tolerance  atol=1.0e-04 rtol=1.0e-04
```

If TensorRT building is requested, the header also prints the selected deployment target and the engine output path.

During loading, several warnings or info lines can be normal:

- `timm` deprecation warnings
- `torch.meshgrid` future warnings
- `torchvision` `pretrained` or `weights` deprecation warnings
- LPIPS checkpoint loading messages
- `timm` / Hugging Face fetch messages for pretrained backbones
- `403 Forbidden` on a `HEAD` request followed by a successful safetensors fallback
- warnings about optional fused window kernels not being installed

These messages do not by themselves indicate a failed export.

## 2. Preflight ONNX parity checks

Before writing the final ONNX file, the workflow exports and checks several smaller components to isolate errors early.

The current workflow checks:

- `model.vit preflight`
- `STDiT block preflight`
- one mid-level attention module, either `SwinTransformerBlock` or `SwinAttention`
- `WindowAttention preflight`

Each check runs:

- one captured sample from the real model path when applicable
- several random samples controlled by `--num-samples`
- ONNX Runtime inference with graph optimizations disabled
- output comparison against PyTorch using the configured `atol` and `rtol`

Typical passing output looks like:

```text
[2] Running preflight ONNX parity checks
  Selected mid-level attention check: SwinAttention
  model.vit preflight  PASS  mismatches=0 max=1.234e-05 mean=9.639e-07
  STDiT block preflight  PASS  mismatches=0 max=6.485e-05 mean=5.512e-07
    block_index=0
  SwinAttention preflight  PASS  mismatches=0 max=1.311e-06 mean=4.841e-08
    module=blocks.0.space_attn resolution=(18, 32) channels=768
  WindowAttention preflight  PASS  mismatches=0 max=8.345e-07 mean=5.046e-08
    module=blocks.0.space_attn.attn tokens=24 channels=768
```

How to read this:

- `PASS` means no sample exceeded the configured tolerance
- `mismatches=0` means all tested samples matched closely enough
- `max` is the worst maximum absolute difference seen across samples
- `mean` is the worst mean absolute difference seen across samples

If any preflight check fails, export stops immediately. That is intentional: the workflow is designed to fail early before writing or trusting a final ONNX graph.

## 3. Exporting `model.vit` to ONNX

If all preflight checks pass, the workflow exports the wrapped transformer to the final ONNX file.

Typical output:

```text
[3] Exporting model.vit to ONNX
  Artifact /mnt/models/orbis/logs_wm/orbis_288x512/onnx/last.onnx
```

The exported model uses four named inputs:

- `target_t`
- `context`
- `t`
- `frame_rate`

The exported model has a single named output:

- `output`

The dummy input shapes are derived from the model configuration and CLI values:

- batch size from `--batch-size`
- context frames from `--context-frames`
- target frames from `--target-frames`
- spatial size from `model.vit.input_size`
- channel count from `model.vit.in_channels`

## 4. Final parity checks on the exported graph

After export, the workflow validates the actual saved ONNX artifact twice:

- once with ONNX Runtime graph optimizations disabled
- once with ONNX Runtime graph optimizations enabled

Typical passing output:

```text
[4] Checking exported model parity with graph optimizations disabled and enabled
  final model parity (graph optimizations disabled)  PASS  mismatches=0 max=1.431e-05 mean=1.094e-06
  final model parity (graph optimizations enabled)  PASS  mismatches=0 max=8.821e-06 mean=1.026e-06

 PASS  Export succeeded and final parity checks passed with graph optimizations both disabled and enabled.
```

Success criteria are intentionally asymmetric:

- If parity fails with graph optimizations disabled, the export is treated as failed.
- If parity passes with optimizations disabled but fails with optimizations enabled, the export still succeeds, but the workflow warns that ONNX Runtime optimizations should be disabled when running inference.

This is a practical guardrail: the unoptimized ONNX graph is treated as the correctness baseline.

## 5. Optional TensorRT engine build

If `--build-tensorrt` is provided and the ONNX export passes parity, the workflow can build a TensorRT engine as a final stage.

Typical output looks like:

```text
[5] Building TensorRT engine for deployment target orin
  Artifact /mnt/models/orbis/logs_wm/orbis_288x512/tensorrt/last_orin_fp16.engine
  Backend  trtexec
  Target   orin
  Precision fp16
```

The builder supports two backends:

- `python`: TensorRT Python bindings
- `trtexec`: the TensorRT command-line builder

With `--tensorrt-backend auto`, the workflow prefers Python bindings first and falls back to `trtexec` if the bindings are unavailable.

TensorRT building is intentionally driven by named input shape profiles for `target_t`, `context`, `t`, and `frame_rate`. That keeps the deployment step aligned with the exported interface rather than with one hard-coded model architecture.

## 6. TensorRT rollout validation

Once an engine has been built, you can validate rollout quality with the TensorRT-specific evaluator:

```bash
python evaluate/rollout_device.py \
  --exp_dir logs_wm/orbis_288x512 \
  --engine tensorrt/v100_e2e_generic_fp16.engine \
  --num_videos 8 \
  --num_gen_frames 4 \
  --save_real true
```

This writes generated frames and GIFs under a TensorRT-specific output folder such as:

- `logs_wm/orbis_288x512/gen_rollout_tensorrt/<data_tag>/...`

To compare the TensorRT rollout against a reference rollout, first generate the reference with the same seed and rollout settings, then point `--reference_rollout_dir` at that directory:

```bash
python evaluate/rollout_onnx.py \
  --exp_dir logs_wm/orbis_288x512 \
  --onnx onnx/last.onnx \
  --num_videos 8 \
  --num_gen_frames 4 \
  --seed 42

python evaluate/rollout_device.py \
  --exp_dir logs_wm/orbis_288x512 \
  --engine tensorrt/v100_e2e_generic_fp16.engine \
  --num_videos 8 \
  --num_gen_frames 4 \
  --seed 42 \
  --reference_rollout_dir gen_rollout_onnx/default_data/ep0iter0_30steps
```

The TensorRT evaluator writes `rollout_report.json` into the output folder. When `--reference_rollout_dir` is provided, the report also includes frame-wise comparison aggregates:

- mean absolute error
- mean squared error
- mean PSNR in dB
- number of compared and missing reference frames

## Warnings you should expect

A clean export can still emit tracing warnings like these during preflight or final export:

```text
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect.
TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect.
```

In this repository, these are commonly triggered by Swin-related shape checks and window bookkeeping during tracing. They are worth noting, but they are not automatically a failure. The parity checks are the real acceptance test.

For the captured successful run, those tracer warnings were present and the exported model still passed all final parity checks.

## Interpreting your current successful run

For the run:

```bash
python export/export_to_device.py --run-dir logs_wm/orbis_288x512/
```

the important outcome is:

- all preflight checks passed
- the ONNX artifact was written to `logs_wm/orbis_288x512/onnx/last.onnx`
- the final exported model passed parity with ORT graph optimizations both disabled and enabled

The representative error magnitudes were small relative to the configured tolerance:

- final parity without graph optimizations: `max=1.431e-05`, `mean=1.094e-06`
- final parity with graph optimizations: `max=8.821e-06`, `mean=1.026e-06`

With `atol=1e-4` and `rtol=1e-4`, that run should be treated as a valid export.

## Failure modes

Common failure categories:

- missing `config.yaml` or checkpoint under the selected run directory
- a model submodule failing parity during preflight
- ONNX export failing because of an unsupported op or shape assumption
- final ONNX artifact loading in ORT but diverging numerically from PyTorch
- TensorRT engine building failing because neither TensorRT Python bindings nor `trtexec` are available
- TensorRT engine building failing because the chosen optimization profile does not cover the intended input shapes

If the script aborts before the export step, focus on the first failing preflight module. The debug scripts in this directory are intended for narrowing down those failures further.

Relevant helpers include:

- `export/debug_stdit_block_onnx.py`
- `export/debug_swin_block_onnx.py`
- `export/debug_window_attention_onnx.py`
- `export/debug_temporal_path_onnx.py`
- `export/debug_spatial_path_onnx.py`

## Practical notes

- The workflow uses CPU ONNX Runtime sessions for parity checks.
- If `TK_WORK_DIR` is referenced in the config and not already set, the loader tries to infer it from standard repo locations.
- Optional pretrained components may be loaded or resolved during model construction if they are part of the configured model.
- The exporter currently targets `model.vit`, not a full end-to-end deployment package.
- TensorRT building is optional and only runs when `--build-tensorrt` is provided.
- TensorRT engine validation is currently a build-time step only; ONNX parity remains the main correctness gate.

## Recommended acceptance checklist

Treat an export as good when all of the following are true:

- the correct run directory, config, and checkpoint are printed at startup
- all preflight checks show `PASS`
- the expected ONNX artifact path is printed in stage 3
- final parity with graph optimizations disabled shows `PASS`
- ideally, final parity with graph optimizations enabled also shows `PASS`
- if TensorRT is requested, the expected engine artifact path is printed in stage 5 and the build completes without backend errors

If you need to debug a failing export, start from the first failing preflight component rather than from the final ONNX artifact.