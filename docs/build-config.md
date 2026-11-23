# Build configuration

The argument-values file (`mdlc -a model.yml`, `mdl_add_model(... CONFIG model.yml)`) holds a model's hyperparameters
and how it is built. One file can describe several builds of the same model, e.g. for inference and for training:

```yaml
args:                                    # hyperparameters per module
  BertForSequenceClassification:
    conf: { dims: { model: 768, key: 64, value: 64, inner: 3072 }, numEncoders: 12, numAttnHeads: 12 }
    emb: { vocab: 30522, positions: 512, types: 2 }
    classes: 3

build:                                   # settings of every build
  ellipsis-rank: { default: 2, labels: 1 }
  float: bf16
  mask: [logits]
  diagnostics: { float-narrowing: warning }

builds:                                  # named builds: `mdlc --build train`, `mdl_add_model(... BUILD train)`
  inference: {}
  train:
    mask: [logits, loss]
    grad: [loss]
    trainable: ["**", "!bert.embeddings.**"]
    checkpoint: ["bert.encoder.layers.*"]
```

## Hyperparameters
`args:` maps module names to the values of their hyperparameters: numbers, booleans, structs as maps, and arrays
(`array[T; n]`) as lists. Hyperparameters that are left out take their default values. A module is compiled if `args:`
has an entry for it (`{}` if all hyperparameters have defaults). For compatibility, the modules may also be at the top
level of the file (without `args:` and without build settings).

## Build settings
The keys of `build:` and of a build in `builds:` are the long names of `mdlc`'s options, with the same values; options
that can be repeated take a list:

| Key | Option | Example |
|---|---|---|
| `ellipsis-rank` | `--ellipsis-rank` | `2`, or `{ default: 2, labels: 1 }` for `--ellipsis-rank 2 --ellipsis-rank labels=1` |
| `float` | `--float` | `bf16` |
| `mask` | `--mask` | `[forward]` |
| `grad` | `--grad` | `[loss]` |
| `trainable` | `--trainable` | `["layers.**", "!layers.0.**"]` |
| `checkpoint` | `--checkpoint` | `["encoder.layers.*"]` |
| `grad-fused`, `grad-overwrite`, `input-grads` | the flags | `true` |
| `kernels` | `--kernels` | `tvm` |
| `fast-math`, `profile` | the flags | `true` |
| `bindings` | `--binding` | `[cpp, python-torch]` |
| `quantize` | `--quantize` | `[{ weights: ["**"], format: int4, group: 128 }]` (see [Quantization](#quantization)) |
| `strict` | `--strict` | `true`: what may be wrong is an error (code that fails for some hyperparameter values, `permits-illegal-instantiation`) |
| `diagnostics` | `-W` | `{ float-widening: warning, unknown-annotation: error, permits-illegal-instantiation: error }` (`error`, `warning`, or `ignored`; the names: [Diagnostics](language-spec.md#diagnostics)) |

The settings of the selected build replace those of `build:` key by key (e.g. `mask` above). Unknown keys, invalid
values, and unknown builds are errors.

## Quantization
`quantize:` stores weight matrices in fewer bits; the model is unchanged. Each rule selects weights by path patterns
(as `trainable`), and later rules take precedence:

```yaml
build:
  quantize:
    - { weights: ["bert.encoder.**"], format: int4, group: 128 }
    - { weights: ["classifier.**"], format: int8 }
```

Only matrices are quantized: a pattern may select whole modules, and their biases and norms stay as they are. A
pattern that selects no matrix (e.g. a typo) is a warning, and a pattern that is the exact path of a weight that is no
matrix (e.g. `layers.0.b`) is an error. A rule needs at least one pattern that is not negated. On the command line, a rule is `--quantize FORMAT[:GROUP]=PATTERN[,PATTERN...]`,
e.g. `--quantize 'int4:128=bert.encoder.**'`.

| `format` | Values | `W.qweight` | Default `group` |
|---|---|---|---|
| `int8` | −127…127 | `int8[N, K]` | `K` (one scale per output) |
| `int4` | −8…7 | `int8[N, K/2]`, two values per byte (low nibble: even k) | 128 |
| `int4-asym` | 0…15, minus a zero point | as `int4` | 128 |

A weight `W: float[N, K]` (outputs × inputs, as in `Linear`) becomes the weights `W.qweight`, `W.scales` (`[K/group, N]`,
the model's float type), and for `int4-asym` `W.zeros` (`int8[K/group, N]`), with
`W[n, k] = scales[k/group, n] · (q[n, k] − zeros[k/group, n])`. This is the layout of GPTQ/AWQ checkpoints (transposed
scales) and of oneDNN's weight decompression. `group` must divide `K` (and be even for 4 bits).

Quantized weights are frozen: gradients skip them, and `trainable` patterns that select them are an error (exclude
them, e.g. `trainable: ["**", "!bert.encoder.**"]`). Quantized checkpoints are loaded as they are; a float checkpoint
is quantized while loading with `mdl::runtime::loadWeights<Model>(path, {}, FloatConversion::none,
Quantization::fromFloat)` (round to nearest, one scale per group from its largest magnitude, or its range for
`int4-asym`).

With the oneDNN kernels, a linear layer reads its quantized weights directly (oneDNN's weight decompression), so the
weights stay compressed in memory. This pays off when reading the weights dominates, i.e. for few tokens per call:
BERT-base with 8 tokens takes 5.4 instead of 10.2 ms with int8 or int4 weights in float32 (4.2 instead of 5.2 ms in
bfloat16), and its weights take 86 or 46 instead of 340 MB. For large batches (1024 tokens) the time is unchanged.

## Precedence
Options on the command line take precedence over the file: an option given there replaces the file's setting, also a
list (`--mask f` replaces `mask: [...]`). So the file holds the defaults, and a single compilation can deviate from
them without editing it. With `mdl_add_model`, the `OPTIONS` are on the command line; note that the target always
passes `--binding cpp=...` (for its header), which replaces a `bindings` setting: request further bindings with
`BINDINGS`.
