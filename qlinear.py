import torch
import torch.nn as nn
import qgemm_cuda

class QuantLinear(nn.Module):
    # ✅ Step 1: Modify __init__ to accept the original layer's data
    def __init__(self, in_features, out_features, bias=True, original_weight=None, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Don't create new weights, use the provided ones
        if original_weight is not None:
            self.weight = nn.Parameter(original_weight.contiguous())
        else:
            # Still allow for creating a new layer from scratch if needed
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            if original_bias is not None:
                self.bias_param = nn.Parameter(original_bias.contiguous())
            else:
                self.bias_param = nn.Parameter(torch.empty(out_features))
                # Proper bias initialization
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / fan_in**0.5 if fan_in > 0 else 0
                nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.register_parameter("bias_param", None)
            
        # ✅ Step 2: Create buffers to store the quantized weights and scales
        self.register_buffer('Wq', torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer('row_scales', torch.empty(out_features, dtype=torch.float32))
        
        # ✅ Step 3: Don't quantize immediately - wait until we're on the right device
        self._weights_quantized = False

    def quantize_weights(self):
        # ✅ This method is called only once to prepare the weights
        with torch.no_grad():
            # Ensure weights are on CUDA before quantization
            if not self.weight.is_cuda:
                self.weight = self.weight.cuda()
            if self.bias_param is not None and not self.bias_param.is_cuda:
                self.bias_param = self.bias_param.cuda()
            
            self.row_scales.copy_(qgemm_cuda.absmax_rowwise(self.weight))
            self.Wq.copy_(qgemm_cuda.quantize_weights_rowwise(self.weight, self.row_scales))
            self._weights_quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ Quantize weights on first forward pass if not already done
        if not self._weights_quantized:
            self.quantize_weights()
            
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
        
        X = x.contiguous()

        # --- Quantize activations (this is correct, as they change) ---
        Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(X.t().contiguous())

        # --- Run quantized GEMM using the pre-quantized weights ---
        Y = qgemm_cuda.qgemm(self.Wq, self.row_scales, Xq, col_scales, col_zps,
                            self.out_features, self.in_features, X.size(0))

        Y = Y.t()

        if self.bias_param is not None:
            Y = Y + self.bias_param

        if len(original_shape) == 3:
            Y = Y.view(original_shape[0], original_shape[1], -1)

        return Y