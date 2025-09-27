import torch
import qgemm_cuda
torch.manual_seed(0)
device = "cuda"

# tiny dims -> easier to inspect
M, K, N = 4, 6, 3

# Draw small values so prints are easy
W = torch.randn(M, K, device=device) * 2.0
X = torch.randn(N, K, device=device) * 2.0   # note: X rows = batch = N, columns = K

# CPU reference: PyTorch Linear convention Y = X @ W.T -> [N, M]
Y_ref = (X @ W.t()).contiguous()

# quantize weights (rowwise on W: rows=M)
row_scales = qgemm_cuda.absmax_rowwise(W)
Wq = qgemm_cuda.quantize_weights_rowwise(W, row_scales)

# For activations we expect Xq shape [K, N] per your wrapper; so transpose before quant.
Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(X.t().contiguous())
# Xq: [K, N]

# Run kernel (Wq: [M,K], Xq: [K,N]) -> Y_q returns [M, N], then we transpose to [N,M]
Y_q = qgemm_cuda.qgemm(Wq, row_scales, Xq.to(torch.uint8).contiguous(), col_scales, col_zps, M, K, N)
Y_q = Y_q.t().contiguous()

print("W (float):\n", W.cpu().numpy())
print("Wq (int8):\n", Wq.cpu().numpy())
print("row_scales:\n", row_scales.cpu().numpy())
print("---")
print("X (float):\n", X.cpu().numpy())
print("Xq (uint8) shape [K,N]:\n", Xq.cpu().numpy())
print("col_scales:\n", col_scales.cpu().numpy())
print("col_zps:\n", col_zps.cpu().numpy())
print("---")
print("Y_ref [N,M] = X @ W.T :\n", Y_ref.cpu().numpy())
print("Y_q  [N,M] = kernel result :\n", Y_q.cpu().numpy())
print("diff (Y_ref - Y_q):\n", (Y_ref - Y_q).cpu().numpy())
print("MSE GEMM:", torch.mean((Y_ref - Y_q)**2).item())