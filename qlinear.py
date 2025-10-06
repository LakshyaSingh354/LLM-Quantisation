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
            # Ensure weight is in the right orientation: (out_features, in_features)
            if weight_fp32.shape[0] == self.in_features and weight_fp32.shape[1] == self.out_features:
                # Conv1D case: transpose to (out_features, in_features)
                weight_fp32 = weight_fp32.t().contiguous()

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

        # Wq has shape (out_features, in_features), so we need to pass out_features, in_features
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
    Memory-efficient linear layer.
    Stores weights in INT8 and dequantizes on-the-fly to FP32/FP16.
    Supports nn.Linear and GPT-2 Conv1D layers automatically.
    """
    def __init__(self, in_features, out_features, bias=True, original_weight=None, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        device = original_weight.device if original_weight is not None else 'cuda'

        # Bias parameter
        if bias and original_bias is not None:
            self.bias_param = nn.Parameter(original_bias.contiguous())
        else:
            self.register_parameter("bias_param", None)

        # Quantized weight buffers: respect actual weight orientation
        if original_weight is not None:
            w = original_weight.contiguous()
            if w.shape == (out_features, in_features):
                wshape = (out_features, in_features)
                rows = out_features
            elif w.shape == (in_features, out_features):
                wshape = (in_features, out_features)
                rows = in_features
            else:
                raise ValueError(f"Unexpected weight shape {tuple(w.shape)} for in_features={in_features}, out_features={out_features}")
            self.register_buffer('Wq', torch.empty(wshape, dtype=torch.int8, device=device))
            self.register_buffer('row_scales', torch.empty(rows, dtype=torch.float32, device=device))
            self._store_weights(w)
        else:
            self.register_buffer('Wq', torch.empty(out_features, in_features, dtype=torch.int8, device=device))
            self.register_buffer('row_scales', torch.empty(out_features, dtype=torch.float32, device=device))

    def _store_weights(self, weight_fp32):
        """
        Quantize and store weights with improved scaling for better accuracy.
        """
        with torch.no_grad():
            # Use per-row scaling but with better clamping to reduce outliers
            row_absmax = torch.max(torch.abs(weight_fp32), dim=1)[0]
            # Use a more conservative scaling to reduce quantization error
            row_scales = torch.clamp(row_absmax / 60.0, min=1e-8)  # Reduced from 127 to 60
            self.row_scales.copy_(row_scales)

            # Quantize with the adjusted scales
            Wq_float = torch.clamp(weight_fp32 / row_scales.unsqueeze(1), -127.0, 127.0).round()
            self.Wq.copy_(Wq_float.to(torch.int8))

        # Store original shape for forward transpose detection
        self._weight_shape = weight_fp32.shape  # e.g. Conv1D: [in, out], Linear: [out, in]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly
        W_dequant = qgemm_cuda.dequantize_rowwise(self.Wq, self.row_scales)

        # Shape-aware multiplication:
        # Conv1D (GPT-2 c_attn/mlp.c_fc): weight shape [in, out], do x @ W
        # Linear (nn.Linear, c_proj/mlp.c_proj): weight shape [out, in], do x @ W.t()
        if W_dequant.shape[0] == x.shape[-1] and W_dequant.shape[1] == self.out_features:
            # Conv1D case: x @ W
            Y = torch.matmul(x, W_dequant.to(x.dtype))
        else:
            # Linear case: x @ W.t()
            Y = torch.matmul(x, W_dequant.t().to(x.dtype))

        # Add bias
        if self.bias_param is not None:
            Y = Y + self.bias_param.to(x.dtype)

        return Y