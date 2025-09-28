import torch
import torch.nn as nn
import qgemm_cuda

class QuantLinear(nn.Module):
    """
    A fully quantized linear layer that replaces a standard nn.Linear.
    It quantizes weights upon initialization and frees the original FP32 weights.
    Activations are quantized on-the-fly during the forward pass.
    """
    def __init__(self, in_features, out_features, bias=True, original_weight=None, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # The device should be inferred from the incoming weight tensor
        device = original_weight.device if original_weight is not None else 'cuda'

        # Bias is small, so we just keep it as a standard parameter
        if bias:
            if original_bias is not None:
                self.bias_param = nn.Parameter(original_bias.contiguous())
            else:
                self.bias_param = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias_param", None)
            
        # ✅ Create buffers on the correct device to store the quantized weights and scales
        # Buffers are part of the module's state but are not considered model parameters.
        self.register_buffer('Wq', torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer('row_scales', torch.empty(out_features, dtype=torch.float32, device=device))
        
        # Perform the quantization and then discard the original FP32 weight
        if original_weight is not None:
            self.quantize_weights(original_weight.contiguous())
        
        # By not assigning `original_weight` to `self.weight`, the FP32 tensor
        # is freed from memory as soon as the `__init__` method completes.

    def quantize_weights(self, weight_fp32):
        """
        Performs per-row absmax quantization on the FP32 weight tensor
        and stores the results in the INT8 buffer. The original FP32 tensor
        is not stored.
        """
        with torch.no_grad():
            # Calculate scales and quantize the weights
            self.row_scales.copy_(qgemm_cuda.absmax_rowwise(weight_fp32))
            self.Wq.copy_(qgemm_cuda.quantize_weights_rowwise(weight_fp32, self.row_scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the quantized matrix multiplication.
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
        
        X = x.contiguous()
        
        # --- Quantize activations on-the-fly (asymmetric, per-column) ---
        Xq, col_scales, col_zps = qgemm_cuda.quantize_activations_colwise(X.t().contiguous())

        # --- Run the highly optimized quantized GEMM kernel ---
        # We use the pre-quantized weights stored in our buffers
        Y = qgemm_cuda.qgemm(self.Wq, self.row_scales, Xq, col_scales, col_zps,
                            self.out_features, self.in_features, X.size(0))

        # Transpose and add bias
        Y = Y.t()
        if self.bias_param is not None:
            Y = Y + self.bias_param

        # Reshape back to original dimensions if necessary
        if len(original_shape) == 3:
            Y = Y.view(original_shape[0], original_shape[1], -1)

        return Y