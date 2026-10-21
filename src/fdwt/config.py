"""Global switches for the precision of the tensor-core (``tl.dot``) kernels.

By default every kernel computes in full fp32 precision, so that results match the fp64
reference to fp32 rounding.

``allow_tf32``: allow TF32 tensor cores for fp32 inputs (~1e-3 relative error, like
cuDNN convolutions under ``torch.backends.cudnn.allow_tf32``).

``allow_bf16``: allow bf16 tensor-core operands with fp32 accumulation for fp32 inputs
(~3e-3 relative error, 2x the TF32 throughput on consumer GPUs).  This mode is also used
automatically for bf16 / fp16 inputs, like ``torch.autocast`` would.

Elementwise work, accumulations and outputs stay in fp32 in every mode; fp64 inputs are
always computed exactly.
"""

allow_tf32: bool = False
allow_bf16: bool = False

# Backward of the fused spectral softmax (``spectral_softmax``, ``MorletAttention``, ...), per
# octave of the pyramid; True / False, or "auto" (by the octave's filter length).
#
# ``fused_input_grad``: scatter dL/dx from the softmax backward kernel (adjoint of its Toeplitz
# tiles, atomic adds) instead of writing dL/dW, (rows, S, N, 2), to memory for the grad-x
# kernel.  "auto": for filters of at most ``fused_input_grad_max_taps`` taps.
#
# ``fused_param_grad``: compute the gradients of the learnable scales and wave number inside the
# same kernel, as inner products with the convolutions by the filter tangents dH/dtheta, instead
# of forming dL/dH with the grad-h kernel (only for the scales / w0 parametrisation, see
# ``cwt.spectral_softmax_params``; an explicit filter tensor always gets dL/dH).  "auto": for
# filters of more than ``fused_param_grad_min_taps`` taps.
#
# Defaults measured on an A100 (REVE EEG shape, 16 frequencies): the grad-h kernel is slow on
# long filters, the separate kernels win on short (replanned) ones.  On an RTX 4080 SUPER,
# fusing both everywhere is ~7% faster for short filters (``fused_input_grad = True``,
# ``fused_param_grad = True``).
fused_input_grad = False
fused_input_grad_max_taps: int = 33
fused_param_grad = "auto"
fused_param_grad_min_taps: int = 33
