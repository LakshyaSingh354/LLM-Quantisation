import torch
import torch.nn as nn
import qgemm_cuda

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_param", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape from [batch, seq_len, in_features] to [batch*seq_len, in_features]
            x = x.view(-1, x.size(-1))
        
        # x: [batch*seq_len, in_features] or [batch, in_features]
        # W: [out_features, in_features]

        W = self.weight.contiguous()
        X = x.contiguous()

        # --- Quantize weights rowwise ---
        row_scales = qgemm_cuda.absmax_rowwise(W)
        Wq = qgemm_cuda.quantize_weights_rowwise(W, row_scales)

        # --- Quantize activations colwise ---
        # Transpose X so that colwise quantization happens along batch dimension
        Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(X.t().contiguous())

        # --- Run quantized GEMM ---
        # Xq is now [in_features, batch*seq_len] after transpose
        Y = qgemm_cuda.qgemm(Wq, row_scales, Xq, col_scales, col_zps,
                            W.size(0), W.size(1), X.size(0))

        # qgemm returns [M, N] = [out_features, batch*seq_len]
        Y = Y.t()  # -> [batch*seq_len, out_features]

        if self.bias_param is not None:
            Y = Y + self.bias_param

        # Reshape back to original shape if input was 3D
        if len(original_shape) == 3:
            # Reshape from [batch*seq_len, out_features] to [batch, seq_len, out_features]
            Y = Y.view(original_shape[0], original_shape[1], -1)

        return Y