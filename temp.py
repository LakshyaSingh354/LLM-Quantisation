import torch
import qgemm_cuda

W = torch.randn(32, 64, device="cuda")
row_scales = qgemm_cuda.absmax_rowwise(W)
Wq = qgemm_cuda.quantize_weights_rowwise(W, row_scales)
W_deq = Wq.float() * row_scales.unsqueeze(1)
print("MSE W:", torch.mean((W - W_deq)**2).item())

X = torch.randn(16, 64, device="cuda")
# Transpose X before quantizing so we get scales per batch item (N=16)
Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(X.t().contiguous())
# Xq is now [64, 16], col_scales and col_zps are [16]
X_deq = (Xq.t().float() - col_zps.unsqueeze(1)) * col_scales.unsqueeze(1)  # Broadcast correctly
print("MSE X:", torch.mean((X - X_deq)**2).item())

Y_ref = W @ X.t()   # [out, batch]
Y_ref = Y_ref.t()   # [batch, out]

# Xq is already [K, N] = [64, 16] as expected by qgemm
Yq = qgemm_cuda.qgemm(Wq, row_scales, Xq.to(torch.uint8).contiguous(), col_scales, col_zps,
                     W.size(0), W.size(1), X.size(0))
Yq = Yq.t()

print("MSE GEMM:", torch.mean((Y_ref - Yq)**2).item())