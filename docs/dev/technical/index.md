\page dev_technical Technical documentation

The following pages describe how the compiler and the tools are built, which design decisions we took, and why.

<div class="cards">

- \subpage dev_pipeline
  From the source text to the object file: elaboration, graph passes, and the TVM backend.
- \subpage dev_optimization
  The general optimizations from the computation graph to the generated code: fusion, save or recompute.
- \subpage dev_cuda_kernels
  The CUDA target: schedule rules of the generated kernels, and the cuBLAS and cuDNN packages.
- \subpage dev_documentation
  How this documentation is built, including the API pages of the standard library.

</div>

\todo Pages on the gradient pass, the masking pass, import resolution, and kernel selection with runtime guards.
