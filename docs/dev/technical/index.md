\page dev_technical Technical documentation

The following pages describe how the compiler and the tools are built, which design decisions we took, and why.

<div class="cards">

- \subpage dev_pipeline
  From the source text to the object file: elaboration, graph passes, and the TVM backend.
- \subpage dev_optimization
  The general optimizations from the computation graph to the generated code: fusion, save or recompute.
- \subpage dev_cuda_kernels
  The CUDA target: schedule rules of the generated kernels, and the cuBLAS, cuDNN and FlashAttention-2 packages.
- \subpage dev_runtime
  How compiled models are called: sessions, the workspace, streams, and CUDA graphs.
- \subpage dev_documentation
  How this documentation is built, including the API pages of the standard library.

</div>

\todo Pages on the gradient pass (beyond what it saves, see \ref dev_optimization), the masking pass, import
resolution, and kernel selection with runtime guards.
