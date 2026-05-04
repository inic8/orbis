import torch
import onnxruntime as ort
import numpy as np
from omegaconf import OmegaConf
from util import instantiate_from_config

CONFIG_PATH = "configs/stage2_myexport.yaml"
CKPT_PATH = "logs_wm/orbis_288x512/checkpoints/last.ckpt"
ONNX_PATH = "logs_wm/orbis_288x512/checkpoints/onnx/last_enhanced.onnx"

print("="*60)
print("Detailed PyTorch vs ONNX Comparison")
print("="*60)

# Load ONNX
print("\n[1] Loading ONNX model...")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
ort_session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options, providers=providers)
print(f"  ✓ Loaded with providers: {ort_session.get_providers()}")

# Load PyTorch
print("\n[2] Loading PyTorch model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = OmegaConf.load(CONFIG_PATH)
model = instantiate_from_config(cfg.model)
state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# Apply EMA weights
print("  Applying EMA weights...")
if hasattr(model, 'ema_vit'):
    ema_params = dict(model.ema_vit.named_parameters())
    for name, param in model.vit.named_parameters():
        if name in ema_params:
            param.data.copy_(ema_params[name].data)

# Ensure determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Test with multiple different inputs
print("\n[3] Running comparison with multiple test cases...")

test_cases = [
    {"seed": 42, "name": "Test 1 (seed=42)"},
    {"seed": 123, "name": "Test 2 (seed=123)"},
    {"seed": 999, "name": "Test 3 (seed=999)"},
]

B, C, H, W = 1, 32, 24, 24
context_frames = 4

results = []

for test in test_cases:
    print(f"\n  {test['name']}:")
    
    # Generate inputs with specific seed
    torch.manual_seed(test['seed'])
    np.random.seed(test['seed'])
    
    target_t = torch.randn(B, 1, C, H, W).to(device)
    context = torch.randn(B, context_frames, C, H, W).to(device)
    t = torch.rand(B).to(device)
    frame_rate = torch.ones(B).to(device)
    
    # PyTorch inference (use vit directly, matching the export wrapper)
    with torch.no_grad():
        pytorch_output = model.vit(target_t, context, t, frame_rate=frame_rate)
    
    # ONNX inference
    onnx_inputs = {
        "target_t": target_t.cpu().numpy(),
        "context": context.cpu().numpy(),
        "t": t.cpu().numpy(),
        "frame_rate": frame_rate.cpu().numpy(),
    }
    onnx_output = ort_session.run(None, onnx_inputs)[0]
    
    # Compare
    pytorch_np = pytorch_output.cpu().numpy()
    diff = np.abs(pytorch_np - onnx_output)
    rel_diff = diff / (np.abs(pytorch_np) + 1e-8)
    
    result = {
        "mean_abs": diff.mean(),
        "max_abs": diff.max(),
        "mean_rel": rel_diff.mean(),
        "max_rel": rel_diff.max(),
        "close_pct": 100 * np.isclose(pytorch_np, onnx_output, rtol=1e-3, atol=1e-5).mean(),
        "pytorch_range": (pytorch_np.min(), pytorch_np.max()),
        "onnx_range": (onnx_output.min(), onnx_output.max()),
    }
    results.append(result)
    
    print(f"    Mean abs diff: {result['mean_abs']:.6f}")
    print(f"    Max abs diff:  {result['max_abs']:.6f}")
    print(f"    Close matches: {result['close_pct']:.2f}%")
    print(f"    PyTorch range: [{result['pytorch_range'][0]:.4f}, {result['pytorch_range'][1]:.4f}]")
    print(f"    ONNX range:    [{result['onnx_range'][0]:.4f}, {result['onnx_range'][1]:.4f}]")

# Summary
print("\n[4] Summary across all test cases:")
avg_mean_abs = np.mean([r['mean_abs'] for r in results])
avg_max_abs = np.mean([r['max_abs'] for r in results])
avg_close_pct = np.mean([r['close_pct'] for r in results])

print(f"  Average mean abs diff: {avg_mean_abs:.6f}")
print(f"  Average max abs diff:  {avg_max_abs:.6f}")
print(f"  Average close matches: {avg_close_pct:.2f}%")

# Detailed analysis of one case
print("\n[5] Detailed analysis of Test 1:")
torch.manual_seed(42)
np.random.seed(42)

target_t = torch.randn(B, 1, C, H, W).to(device)
context = torch.randn(B, context_frames, C, H, W).to(device)
t = torch.rand(B).to(device)
frame_rate = torch.ones(B).to(device)

with torch.no_grad():
    pytorch_output = model.vit(target_t, context, t, frame_rate=frame_rate)

onnx_inputs = {
    "target_t": target_t.cpu().numpy(),
    "context": context.cpu().numpy(),
    "t": t.cpu().numpy(),
    "frame_rate": frame_rate.cpu().numpy(),
}
onnx_output = ort_session.run(None, onnx_inputs)[0]

pytorch_np = pytorch_output.cpu().numpy()
diff = np.abs(pytorch_np - onnx_output)

# Analyze spatial pattern of differences
print("\n  Spatial distribution of errors:")
diff_per_spatial = diff.reshape(B, -1, H, W).mean(axis=1)  # Average across channels
print(f"    Shape after spatial averaging: {diff_per_spatial.shape}")
print(f"    Max error location: {np.unravel_index(diff_per_spatial.argmax(), diff_per_spatial.shape)}")

# Analyze channel-wise differences
diff_per_channel = diff.reshape(B, -1, H*W).mean(axis=2)  # Average across spatial
print(f"\n  Channel-wise errors:")
print(f"    Shape: {diff_per_channel.shape}")
print(f"    Min channel error: {diff_per_channel.min():.6f}")
print(f"    Max channel error: {diff_per_channel.max():.6f}")
print(f"    Std of channel errors: {diff_per_channel.std():.6f}")

# Check for systematic bias
bias = (pytorch_np - onnx_output).mean()
print(f"\n  Systematic bias: {bias:.6f}")
if abs(bias) > 0.01:
    print("    ⚠ Significant bias detected - ONNX outputs are systematically different")
else:
    print("    ✓ No significant systematic bias")

# Percentile analysis
print(f"\n  Error percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"    {p}th percentile: {np.percentile(diff, p):.6f}")

# Final assessment
print("\n" + "="*60)
print("Assessment:")
print("="*60)

if avg_max_abs < 1e-4:
    print("✓ EXCELLENT: Outputs match very closely")
    print("  The differences are within expected numerical precision.")
elif avg_max_abs < 1e-3:
    print("✓ GOOD: Outputs match acceptably")
    print("  Small differences may be due to numerical precision or minor op differences.")
elif avg_max_abs < 1e-2:
    print("⚠ MARGINAL: Outputs have noticeable differences")
    print("  This may be acceptable depending on your use case.")
    print("  Consider investigating further.")
else:
    print("✗ POOR: Outputs differ significantly")
    print("  There are likely fundamental differences in how operations are computed.")
    print("\n  Possible causes:")
    print("  1. Operations not supported or implemented differently in ONNX")
    print("  2. Numerical instability in certain operations")
    print("  3. Different handling of edge cases (NaN, Inf, etc.)")
    print("  4. Custom CUDA kernels not available in ONNX Runtime")
    print("\n  Recommended actions:")
    print("  - Run diagnose_onnx_issue.py to identify problematic operations")
    print("  - Check if model uses custom attention mechanisms")
    print("  - Try simplifying the model architecture")
    print("  - Consider using TorchScript instead of ONNX")