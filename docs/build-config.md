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
| `diagnostics` | `-W` | `{ float-widening: warning, unknown-annotation: error }` (`error`, `warning`, or `ignored`) |

The settings of the selected build replace those of `build:` key by key (e.g. `mask` above). Unknown keys, invalid
values, and unknown builds are errors.

## Precedence
Options on the command line take precedence over the file: an option given there replaces the file's setting, also a
list (`--mask f` replaces `mask: [...]`). So the file holds the defaults, and a single compilation can deviate from
them without editing it. With `mdl_add_model`, the `OPTIONS` are on the command line; note that the target always
passes `--binding cpp=...` (for its header), which replaces a `bindings` setting: request further bindings with
`BINDINGS`.
