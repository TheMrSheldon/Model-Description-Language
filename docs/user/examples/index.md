\page user_examples Examples and tutorials
\tableofcontents

The directory `examples/` contains complete projects, each with a model, its build configuration, and a C++ (or
Python) program that uses the compiled model. They build with the main project (`MDL_BUILD_EXAMPLES`) and run as tests,
so they always work with the current version. The pages below walk through them; the code shown there is taken from the
example files when the documentation is built.

# Walkthroughs
<div class="cards">

- \subpage example_ffn
  The smallest complete project: a model, its configuration, and a C++ program that calls it.
- \subpage example_gradients
  Compiling gradients of a model and of a loss, and checking them.

</div>

# All examples
| Example | What it shows |
|---|---|
| \ref example_ffn "01_ffn" | a feed-forward layer: the smallest complete project |
| `02_transformer` | a BERT-style transformer encoder \cite vaswani2017attention \cite devlin2019bert with a pooler and sequence classification |
| `03_pytorch` | using a compiled model from PyTorch |
| \ref example_gradients "04_gradients" | compiling gradients (`--grad`) of a model and of a loss |
| `05_classification` | a classifier with a loss and its training loop |
| `06_masking` | masked tensors for batches of sequences with different lengths |
| `07_pytorch_training` | training a compiled model from PyTorch |
| `08_hyperparameters` | hyperparameters, defaults, and named builds |
| `09_quantization` | quantized weights (int4/int8) |

\todo Walkthroughs of the other examples, in the style of \ref example_ffn.
