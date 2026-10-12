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
