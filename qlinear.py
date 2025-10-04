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
        
        # ✅ --- Use the NEW Clipped Activation Quantization Kernel ---
        # Note: We are assuming your new kernel is named `quantize_activations_colwise_clipped`
        # For simplicity, we pass the single min/max to all columns.
        Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(
            X.t().contiguous(), self.calib_min, self.calib_max
        )

        Y = qgemm_cuda.qgemm(self.Wq, self.row_scales, Xq, col_scales, col_zps,
                            self.out_features, self.in_features, X.size(0))

        Y = Y.t()
        if self.bias_param is not None:
            Y = Y + self.bias_param

        if len(original_shape) == 3:
            Y = Y.view(original_shape[0], original_shape[1], -1)

        return Y