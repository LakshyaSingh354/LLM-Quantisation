import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from qlinear import QuantLinear, QuantLinearDequant
import math, time, gc
from transformers.pytorch_utils import Conv1D

model_name = "gpt2-large"
device = "cuda"

model = GPT2LMHeadModel.from_pretrained(model_name).eval().to(device)

layers_to_replace = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, Conv1D)):
        layers_to_replace.append(name)
print(f"Found {len(layers_to_replace)} layers")

bad_layers = []

def test_one_orientation(name, orig_mod, w_for_quant_cpu, b_cpu, transpose_flag):
    """
    Build a QuantLinearDequant with w_for_quant_cpu (CPU tensor),
    moving it to device during construction, then test and return diagnostics.
    transpose_flag: whether w_for_quant_cpu is already transposed (True means w_cpu was transposed)
    """
    # Move weight/bias to device to construct replacement
    w_dev = w_for_quant_cpu.to(device)
    b_dev = b_cpu.to(device) if b_cpu is not None else None

    if isinstance(orig_mod, Conv1D):
        in_features = orig_mod.weight.shape[1]
    elif isinstance(orig_mod, nn.Linear):
        in_features = orig_mod.in_features
    else:
        raise ValueError(f"Unknown module type: {type(orig_mod)}")

    # Small random input
    xt = torch.randn(4, in_features, device=device)
    out_features, in_features = w_for_quant_cpu.shape

    try:
        new_mod = QuantLinearDequant(
            in_features, out_features,
            bias=(b_cpu is not None),
            original_weight=w_dev,
            original_bias=b_dev
        ).to(device).eval()
    except Exception as e:
        return {"ok": False, "err": f"construct_failed: {e}"}

    # Build small input

    # Try to call original module directly; fall back to F.linear if necessary
    try:
        with torch.no_grad():
            out_orig = orig_mod(xt)
        fallback = False
    except Exception:
        # fallback: use F.linear with original weight/bias
        # Conv1D does x @ W.T, so we need to transpose weight
        w_orig = orig_mod.weight
        b_orig = orig_mod.bias
        if isinstance(orig_mod, Conv1D):
            w_orig = w_orig.T
        with torch.no_grad():
            out_orig = F.linear(xt, w_orig, b_orig)
        fallback = True


    # new_mod forward
    with torch.no_grad():
        out_new = new_mod(xt)

    # errors
    try:
        diff = (out_orig - out_new).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
    except Exception as e:
        # shape mismatch etc.
        max_abs = float("nan")
        mean_abs = float("nan")

    # check weight reconstruction
    try:
        import qgemm_cuda
        W_dequant = qgemm_cuda.dequantize_rowwise(new_mod.Wq, new_mod.row_scales)
        w_ref = w_dev.to(W_dequant.device)
        w_recon_diff = (w_ref - W_dequant).abs()
        w_recon_max = w_recon_diff.max().item()
        w_recon_mean = w_recon_diff.mean().item()
    except Exception as e:
        w_recon_max = None
        w_recon_mean = None

    # cleanup
    del new_mod, xt, out_orig, out_new
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.01)

    return {
        "ok": True,
        "transpose": transpose_flag,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "w_recon_max": w_recon_max,
        "w_recon_mean": w_recon_mean,
        "fallback_used": fallback
    }

for i, name in enumerate(layers_to_replace):
    orig_mod = model.get_submodule(name)

    # fetch CPU copy of weight/bias (small)
    with torch.no_grad():
        w_cpu = orig_mod.weight.detach().clone().cpu().contiguous()
        b_cpu = orig_mod.bias.detach().clone().cpu().contiguous() if getattr(orig_mod, "bias", None) is not None else None

    # prepare two candidate orientations
    candA_w = w_cpu           # no transpose (assume weight is [out, in])
    candB_w = w_cpu.t().contiguous()  # transpose (assume weight originally [in, out])

    # test both (order: no-transpose first)
    resA = test_one_orientation(name, orig_mod, candA_w, b_cpu, transpose_flag=False)
    resB = test_one_orientation(name, orig_mod, candB_w, b_cpu, transpose_flag=True)

    # select the candidate with the smaller max_abs (prefer valid numeric)
    def score(r):
        if not r.get("ok", False):
            return float("inf")
        m = r.get("max_abs", float("nan"))
        if math.isnan(m):
            return float("inf")
        return m

    scoreA = score(resA)
    scoreB = score(resB)

    chosen = None
    if scoreA <= scoreB:
        chosen = resA
    else:
        chosen = resB

    transpose_note = "(transposed)" if chosen.get("transpose") else ""
    print(f"{i:03d} {name:60s} {transpose_note} | max_abs={chosen.get('max_abs')} mean_abs={chosen.get('mean_abs')} | "
          f"W_recon_max={chosen.get('w_recon_max')} W_recon_mean={chosen.get('w_recon_mean')} | fallback={chosen.get('fallback_used')}")

    if chosen.get("max_abs") is None or (not math.isnan(chosen.get("max_abs")) and chosen.get("max_abs") > 1e-2):
        bad_layers.append((i, name, chosen.get("max_abs"), chosen.get("mean_abs"), chosen.get("transpose")))

    # free CPU copies
    del w_cpu, b_cpu
    torch.cuda.empty_cache()
    gc.collect()

print("BAD LAYERS:", bad_layers)
