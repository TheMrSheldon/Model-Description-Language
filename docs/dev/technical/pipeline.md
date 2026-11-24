\page dev_pipeline The compilation pipeline
\tableofcontents

This page follows a model from its source text to the object file that `mdlc` writes, and explains the main design
decisions along the way.

# Overview
| Stage | Input → output | Code |
|---|---|---|
| Parsing and elaboration | `.dlm` source → computation graph | `src/mdl/parser.cpp`, `src/mdl/mdl.g4` |
| Exports | compilation unit → the functions to compile | `src/mdl/exports.cpp` |
| Graph passes | computation graph → computation graph | `src/mdl/passes/` (shapes, autodiff, masking) |
| Lowering | computation graph → TVM Relax | `src/mdl/backends/tvm/tvm.cpp` |
| Canonicalization, kernel selection | Relax → Relax | `src/mdl/backends/tvm/details/{pipeline,patterns,kernel_match}.cpp` |
| Legalization, scheduling | Relax → TIR | `src/mdl/backends/tvm/details/{legalize,schedule}.cpp` |
| Code generation | TIR → object file | `src/mdl/backends/tvm/details/aot.cpp` |

The backend builds on TVM \cite chen2018tvm.

# Parsing and elaboration
The grammar (`mdl.g4`) is translated by ANTLR. Instead of building an abstract syntax tree first, the parser's visitor
*elaborates* the program directly: calling a module with concrete hyperparameters instantiates it, compile-time loops
are unrolled, and `?:` picks one branch. The result is a computation graph whose nodes are tensor operations.

Why not an AST plus separate passes? MDL's programs are evaluated at compile time to a large degree — hyperparameters
decide the number of layers, the shapes, and which members exist (`@conditional`). Doing name resolution, type checking
and elaboration in one walk keeps these interactions in one place. For the tools, the same walk fills a semantic index
(`mdl::analysis::SemanticIndex`): definitions, references, types and documentation, which the language server and
`mdl-doc` use. Modules with hyperparameters are additionally checked once with *symbolic* values, so that errors are
found for all instantiations, not only the compiled one.

# The computation graph
The graph is flat: a function is a list of nodes that refer to each other, without nested regions. Operations are a
fixed set of builtins (the standard library is written in MDL on top of them), and information that only some passes
need (e.g. masks) lives in side tables rather than in the nodes. This keeps the passes simple and makes it easy for a
backend to recognize patterns (e.g. a softmax) in the graph.

# Graph passes
- **Shapes** infers and checks the shapes of all nodes.
- **Autodiff** generates gradient functions by reverse-mode differentiation (see \ref foundations_autodiff).
- **Masking** rewrites operations on masked tensors (dense values plus a mask of the valid elements) so that padding
  does not leak into reductions.

# The TVM backend
The graph is lowered to TVM's Relax. A canonicalization phase normalizes the graph into the forms that kernel packages
target (see \ref spec_kernel_packages). Kernel selection then replaces matching regions by calls to optimized kernels
(e.g. oneDNN); where a kernel's constraints can only be decided at runtime (e.g. a symbolic size), the region is
guarded by a runtime check with the original computation as fallback. All remaining operations are legalized into TIR
and scheduled by TVM, and the whole program is compiled ahead of time into one object file that exports a function per
compiled member function.

\todo Memory planning, the calling convention, and the bindings.
