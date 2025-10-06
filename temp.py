import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from qlinear import QuantLinear, QuantLinearDequant
import math, time, gc
import qgemm_cuda

model_name = "gpt2-large"
device = "cuda"

# Load model once (single copy)
model = GPT2LMHeadModel.from_pretrained(model_name).eval().to(device)

bad_layers = []

# collect candidate layers (Linear and Conv1D-like)
layers_to_replace = []
for name, module in model.named_modules():
    # Keep your original detection; if you need to detect Conv1D, adjust here.
    if isinstance(module, (nn.Linear, Conv1D)):
        # We include all modules; we'll skip non-linear modules later when needed.
        layers_to_replace.append(name)

print(f"Found {len(layers_to_replace)} candidate names (will filter inside loop)")

# helper: safe device/dtype print
def dbg_device_dtype(t):
    if t is None:
        return "None"
    return f"{t.device}/{t.dtype}/{tuple(t.shape)}"

for i, name in enumerate(layers_to_replace):
    try:
        orig_mod = model.get_submodule(name)
    except Exception as e:
        print(f"{i:03d} {name:60s} | could not get submodule: {e}")
        continue

    # Only test modules that behave like a linear layer (nn.Linear or conv1d-like weights)
    # We will attempt to infer weight tensor and correct orientation.
    if not hasattr(orig_mod, "weight"):
        # Not a weight-bearing module we care about
        continue

    # Clone weights/bias to CPU to keep GPU memory low
    with torch.no_grad():
        w_cpu = orig_mod.weight.detach().clone().cpu().contiguous()
        b_cpu = orig_mod.bias.detach().clone().cpu().contiguous() if getattr(orig_mod, "bias", None) is not None else None

    # Try to infer in/out features and whether a transpose is needed.
    # For HF Conv1D-like modules people often transpose before passing to QuantLinear.
    # We'll detect a likely Conv1D if weight shape is (in, out) rather than (out, in).
    w_shape = tuple(w_cpu.shape)
    # Heuristic: treat (out, in) as standard linear; (in, out) as Conv1D-like that needs transpose
    needs_transpose = False
    if len(w_shape) == 2:
        if w_shape[0] < w_shape[1]:
            # common case: out < in (still could be linear) — don't assume transpose
            needs_transpose = False
        # If weight looks like (in, out) where rows >> cols, detect by checking typical GPT dims:
        # We choose a safe rule: if module is nn.Linear, assume (out,in). If not nn.Linear, try transpose.
        if isinstance(orig_mod, nn.Linear):
            needs_transpose = False
        else:
            # For non-Linear (e.g., transformers.Conv1D) we will attempt transpose.
            needs_transpose = True
    else:
        # Irregular weight shape; skip
        print(f"{i:03d} {name:60s} | skipping (unhandled weight shape {w_shape})")
        continue

    # Build the weight tensor that QuantLinear expects: (out_features, in_features)
    if needs_transpose:
        w_for_quant = w_cpu.t().contiguous()
        # For debugging, note we transposed
        trans_note = "(transposed)"
    else:
        w_for_quant = w_cpu
        trans_note = ""

    out_features, in_features = w_for_quant.shape[0], w_for_quant.shape[1]

    # Move original module to eval and ensure it's on device (it already is since model is on device)
    orig_mod = orig_mod.eval()

    # Build the quantized replacement, moving the small weight/bias to device for construction
    # Note: QuantLinearDequant will quantize the weight and store Wq and row_scales as buffers.
    try:
        new_mod = QuantLinearDequant(
            in_features, out_features,
            bias=(b_cpu is not None),
            original_weight=w_for_quant.to(device),
            original_bias=b_cpu.to(device) if b_cpu is not None else None
        ).to(device).eval()
    except Exception as e:
        print(f"{i:03d} {name:60s} | failed to construct QuantLinearDequant: {e}")
        del w_cpu, b_cpu
        torch.cuda.empty_cache()
        gc.collect()
        continue

    # Build a small random input. Use batch=4 by default.
    # We must ensure the input shape and dtype are appropriate for orig_mod.
    # Most linear-like modules expect input shape (B, in_features).
    xt = torch.randn(4, in_features, device=device)

    # Compute original module output by calling orig_mod directly.
    # Warning: some modules in HF may expect input shapes different than (B, in_f) --
    # but most Linear / Conv1D-like layers accept (B, in_f).
    try:
        with torch.no_grad():
            out_orig = orig_mod(xt.to(next(orig_mod.parameters()).device))
    except Exception as e:
        # If direct call to orig_mod fails (e.g. expects different shape), try F.linear with w_cpu
        try:
            out_orig = F.linear(xt, w_for_quant.to(device), b_cpu.to(device) if b_cpu is not None else None)
            fallback_note = " (fallback F.linear used)"
        except Exception as e2:
            print(f"{i:03d} {name:60s} | orig_mod(xt) failed and F.linear fallback failed: {e} / {e2}")
            del new_mod, w_cpu, b_cpu, xt
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            fallback_note = " (fallback F.linear used)"
    else:
        fallback_note = ""

    # Compute new module output
    with torch.no_grad():
        out_new = new_mod(xt)

    # Compute errors
    try:
        diff = (out_orig - out_new).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
    except Exception as e:
        print(f"{i:03d} {name:60s} | error computing diff: {e}")
        max_abs = float("nan")
        mean_abs = float("nan")

    # Sanity: compare dequantized weight vs original
    # Use qgemm_cuda.dequantize_rowwise to produce W_dequant and compare with w_for_quant (on-device)
    try:
        import qgemm_cuda
        W_dequant = qgemm_cuda.dequantize_rowwise(new_mod.Wq, new_mod.row_scales)  # should be device tensor
        # Move original reference weight to same device for comparison
        w_ref = w_for_quant.to(W_dequant.device)
        w_recon_diff = (w_ref - W_dequant).abs()
        w_recon_max = w_recon_diff.max().item()
        w_recon_mean = w_recon_diff.mean().item()
    except Exception as e:
        w_recon_max = None
        w_recon_mean = None

    print(f"{i:03d} {name:60s} {trans_note}{fallback_note} | max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} | "
          f"W_recon_max={w_recon_max} W_recon_mean={w_recon_mean} | orig_dev={dbg_device_dtype(next(orig_mod.parameters()))} new_dev={dbg_device_dtype(next(new_mod.parameters()))}")

    # Flag "bad" layers (conservative threshold)
    if not math.isnan(max_abs) and max_abs > 1e-2:
        bad_layers.append((i, name, max_abs, mean_abs))

    # cleanup GPU memory references
    del new_mod, xt, out_orig, out_new
    # w_cpu and b_cpu remain on CPU; remove them now
    del w_cpu, b_cpu
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.01)  # small pause to let the driver clean up

print("BAD LAYERS:", bad_layers)


# Minimum and Maximum max_abs from bad_layers
print("Minimum max_abs:", min(bad_layers, key=lambda x: x[2])[2])
print("Maximum max_abs:", max(bad_layers, key=lambda x: x[2])[2])


w = orig_mod.weight.detach().clone().to(device)
row_scales = qgemm_cuda.absmax_rowwise(w)
wq = qgemm_cuda.quantize_weights_rowwise(w, row_scales)
w_deq = qgemm_cuda.dequantize_rowwise(wq, row_scales)

err = (w - w_deq).abs()
print("mean", err.mean().item(), "max", err.max().item())
