import torch
import torch.nn as nn
import qgemm_cuda 

class QuantLinear(nn.Module):
    """
    A fully quantized linear layer that uses pre-computed calibration ranges
    for robust activation quantization (clipping).
    """
    def __init__(self, in_features, out_features, bias=True, 
                 original_weight=None, original_bias=None, 
                 calib_min_t=None, calib_max_t=None): # ✅ Accept calibration ranges
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        device = original_weight.device if original_weight is not None else 'cuda'

        if bias:
            if original_bias is not None:
                self.bias_param = nn.Parameter(original_bias.contiguous())
            else:
                self.bias_param = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias_param", None)
            
        # --- Weight Quantization (Done Once) ---
        self.register_buffer('Wq', torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer('row_scales', torch.empty(out_features, dtype=torch.float32, device=device))
        if original_weight is not None:
            self.quantize_weights(original_weight.contiguous())

        # ✅ --- Store Calibration Ranges as Buffers ---
        # We need to handle the case of per-column quantization (a tensor of ranges)
        # For now, we assume a single min/max for the whole activation tensor for simplicity.
        if calib_min_t is not None:
            self.register_buffer('calib_min', calib_min_t.clone())
        if calib_max_t is not None:
            self.register_buffer('calib_max', calib_max_t.clone())


    def quantize_weights(self, weight_fp32):
        with torch.no_grad():
            self.row_scales.copy_(qgemm_cuda.absmax_rowwise(weight_fp32))
            self.Wq.copy_(qgemm_cuda.quantize_weights_rowwise(weight_fp32, self.row_scales))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
        
        X = x.contiguous()
        
        # Ensure calibration ranges match expected shape
        # calib_min/max should have shape [in_features] for per-column quantization
        if self.calib_min.dim() == 0:
            # Broadcast scalar to vector
            calib_min = self.calib_min.expand(self.in_features)
            calib_max = self.calib_max.expand(self.in_features)
        else:
            calib_min = self.calib_min
            calib_max = self.calib_max
        
        Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(
            X.t().contiguous(), calib_min, calib_max
        )

        Y = qgemm_cuda.qgemm(self.Wq, self.row_scales, Xq, col_scales, col_zps,
                            self.out_features, self.in_features, X.size(0))

        Y = Y.t()
        if self.bias_param is not None:
            Y = Y + self.bias_param

        if len(original_shape) == 3:
            Y = Y.view(original_shape[0], original_shape[1], -1)

        return Y


class QuantLinearDequant(nn.Module):
    """
    A memory-efficient linear layer.
    Stores weights in INT8 and dequantizes them on-the-fly to FP16/FP32 for computation.
    This preserves perplexity while slashing memory usage.
    """
    def __init__(self, in_features, out_features, bias=True, original_weight=None, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        device = original_weight.device if original_weight is not None else 'cuda'

        # Bias is small, so we keep it in high precision
        if bias and original_bias is not None:
            self.bias_param = nn.Parameter(original_bias.contiguous())
        else:
            self.register_parameter("bias_param", None)
            
        # --- Weight Quantization (Done Once) ---
        # We store the quantized weights and scales as buffers.
        self.register_buffer('Wq', torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer('row_scales', torch.empty(out_features, dtype=torch.float32, device=device))
        
        # Quantize and then discard the original FP32 weight to save memory
        if original_weight is not None:
            self.quantize_weights(original_weight.contiguous())

    def quantize_weights(self, weight_fp32):
        with torch.no_grad():
            # Use your existing CUDA kernels for this one-time operation
            self.row_scales.copy_(qgemm_cuda.absmax_rowwise(weight_fp32))
            self.Wq.copy_(qgemm_cuda.quantize_weights_rowwise(weight_fp32, self.row_scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. On-the-fly dequantize the INT8 weights to a temporary tensor
        W_dequant = qgemm_cuda.dequantize_rowwise(self.Wq, self.row_scales)

        # 2. Use PyTorch's highly optimized matmul.
        # Ensure the dequantized weight matches the activation's dtype (usually FP16 for speed).
        Y = torch.matmul(x, W_dequant.t().to(x.dtype))

        # 3. Add bias if it exists
        if self.bias_param is not None:
            Y = Y + self.bias_param.to(x.dtype)

        return Y