# Kernel packages

Optimized kernels come from **kernel packages** in kernel directories. A package describes what it matches in the
model, how to call its prebuilt implementation (a C function), and which binaries provide it for which platforms. The
compiler selects implementations and falls back to TVM's code generation for everything else.

**Status:** the package format (`schema: 1`), the kernel directories, matching, selection, compiling with the selected
implementations, the link report, and the CMake integration are implemented. Of the built-in oneDNN kernels, softmax
is a package (`kernels/onednn`); the others are still called by the compiler directly. Undecided constraints make an
implementation ineligible (runtime checks are planned); CUDA artifacts are not supported yet.

## Kernel directories
Each immediate subdirectory of a kernel directory that contains a `package.yml` is a package. Its directory must have
the package's name. The directories, most specific first:

1. `--kernels-dir DIR` (repeatable);
2. the project: `.mdl/kernels`, searched upward from the source file;
3. the user's directory: `$XDG_DATA_HOME/mdl/kernels` (default `~/.local/share/mdl/kernels`),
   `%LOCALAPPDATA%\mdl\kernels` on Windows, `~/Library/Application Support/mdl/kernels` on macOS;
4. global: the directories in `MDL_KERNELS_PATH` (separated by `:`, `;` on Windows), the installation's
   `<prefix>/share/mdl/kernels`, and the source tree's `kernels/` (when running from the build tree).

A package in a more specific directory hides a package of the same name in a later one (e.g. a project's own build of
a package). Packages with problems are skipped and reported.

```
$ mdl-kernels list [--kernels-dir DIR]... [model.dlm]
```
lists the directories, the packages with their artifacts (and whether they can be used on this host) and
implementations, the hidden packages, and the problems. The exit code is 1 if a package has problems. Listing never
opens an artifact.

## Matching a model: `mdl-kernels check` and `dump`
```
$ mdl-kernels check [--explain] [--kernels-dir DIR]... model.dlm [mdlc options]
$ mdl-kernels dump model.dlm [mdlc options]
```
Both compile the model like `mdlc` (with the same options, e.g. `-a model.yml --ellipsis-rank 2 --grad forward`) up to
the **matching boundary**: the Relax module after constant folding and the compiler's canonical forms (`relax.nn.*`
operators, `mdl.matmul` with folded permutations), before legalization. No code is generated.

- `dump` prints the module at the boundary: the representation that patterns are written against. (`mdlc --dump-dir`
  writes it too, as `<stem>.relax-match.txt`.)
- `check` lists the selected implementations per function. With `--explain`, it lists every candidate: the bindings it
  would replace, its captures, the verdict of each constraint (proven, not met, or unknown, i.e. undecided at compile
  time), and why it is selected or not:
  ```
  function M__forward
    competing/scaled-add-aligned (priority 20) at v4
      replaces v3, v4; captures $x = x, $y = y
      proven   v3 is used only inside the region
      proven   a usable artifact (host)
      unknown  where $x.shape[0] % 16 == 0 (undecided at compile time: B % 16 == 0) [could be checked at runtime]
      => ineligible: 1 constraint(s) undecided at compile time
  ```

**Selection.** All candidates are collected first (every pattern at every binding), so a small match cannot hide a
larger one. A candidate is eligible if all its constraints are proven; an undecided constraint makes it ineligible
(runtime checks for shape arithmetic are planned). Eligible candidates are selected by priority, then the number of
replaced bindings (larger regions first), then the package and implementation names; a candidate is not selected if it
replaces a binding that a selected one replaces. Captured inputs may be shared.

**Region rules.** The bindings of a region are replaced by one call: an intermediate result that is also used outside
the region makes the candidate ineligible.

## Compiling with kernel packages
`mdlc` matches the packages of the kernel directories at the matching boundary and replaces every selected region by a
call of its implementation: a C function call (`call_extern`) with the evaluated arguments, whose nonzero return value
is a runtime error naming the kernel. Everything else is compiled by TVM as before.

- `--kernel-package NAME` (repeatable) uses only these packages, `--no-kernel-package NAME` excludes one, and
  `--no-kernel-packages` uses none.
- `--kernel-report FILE` writes what the compiled code calls and what must be linked (the structured dependency
  information; integrations translate it):
  ```json
  { "schema": 1,
    "kernels": [ { "function": "M__forward", "package": "scaled_add_c", "version": "1.0",
                   "implementation": "scaled-add-f32", "symbol": "test_scaled_add_f32", "artifact": "host",
                   "manifest": ".../package.yml", "manifest_hash": "fnv1a64:..." } ],
    "link": [ { "package": "scaled_add_c", "artifact": "host", "kind": "static-library", "path": ".../libscaled_add_c.a",
                "hash": "fnv1a64:...", "dependencies": ["system-library:m"] } ] }
  ```
  `link` lists every library in link order (an artifact before the artifacts it depends on).
- The depfile (`--depfile`) also lists the manifests and libraries of the used packages, so changing a package
  rebuilds the model.
- `mdl-kernels link-args REPORT [--style gnu|msvc]` prints linker arguments for a report (for builds without CMake).

### CMake
- `mdl_add_model(... KERNEL_PACKAGES <name>...)` lets the model use these packages and links their libraries (and
  their dependencies) into the model's target, so that they propagate to the executable. Without it, the model uses no
  kernel packages (a package that CMake does not link could not be linked). The `onednn` package is added whenever the
  model uses the oneDNN kernels (`KERNELS onednn`, the default).
- `mdl_kernel_package(<name> MANIFEST package.yml.in SOURCES ...)` builds a package in the project (the library and
  the manifest, into the build tree's kernel directory); with `TARGETS` instead of `SOURCES`, existing targets are the
  libraries (see `kernels/onednn`).
- Packages that the project does not build (e.g. installed ones) are imported at configure time with
  `mdl-kernels deps --kernel-package <name>`: it prints the package's libraries for this host as CMake `IMPORTED`
  targets (`mdl_kernel_package_<name>`) with their link dependencies (`--format json` for other tools).

## `package.yml`
```yaml
schema: 1                   # the format version (required)
package: scaled_add         # the name (= the directory's name)
version: "1.0"
description: Optional text.

artifacts:                  # binaries, per platform
  host-linux:
    kind: static-library    # static-library, shared-library (cubin, ptx: reserved for CUDA)
    path: lib/libscaled_add.a              # relative to the package
    platform: { os: linux, arch: x86_64, abi: gnu, libc: ">=2.28" }
    cpu-features: [avx2]    # optional
    dependencies:           # structured; integrations (CMake, ...) translate them
      - { kind: system-library, name: m }
      - { kind: runtime, name: openmp }
      - { kind: artifact, name: other-artifact }

patterns:                   # what is matched (structured, at the matching boundary)
  scaled-add:
    op: relax.add
    args:
      - op: relax.multiply
        args: [{ capture: x }, { const: 2.0 }]
      - { capture: y }

implementations:
  scaled-add-f32:
    pattern: scaled-add
    artifacts: [host-linux] # the first one that is compatible with the target is used
    priority: 10
    symbol: test_scaled_add_f32            # one symbol per element type
    signature:
      returns: { type: int32, error: nonzero }
      arguments:
        - { type: pointer, access: write, value: $out.data, align: 64 }
        - { type: pointer, access: read, value: $x.data }
        - { type: pointer, access: read, value: $y.data }
        - { type: int64, value: "prod($x.shape)" }
      aliasing: none        # or { out: x }: the output may be x's memory
      outputs:
        - { shape: $x.shape, dtype: $x.dtype }
    applicability:
      dtype: { x: float32, y: float32 }
      layout: { x: row-major, y: row-major }
      where: ["$x.shape == $y.shape"]
    guarantees: { numerics: { exact: true } }  # declared by the author, never derived by the compiler
```
Unknown keys, unknown schema versions, and references to unknown patterns or artifacts are errors (with the file and
line). YAML anchors and merge keys work, e.g. for implementations that differ in the symbol and element type:
```yaml
  softmax-f32: &softmax
    pattern: softmax
    symbol: mdl_softmax_f32
    ...
  softmax-bf16:
    <<: *softmax
    symbol: mdl_softmax_bf16
    applicability: { dtype: { x: bfloat16 } }
```

### Artifacts
- `platform` states binary compatibility: `os` (linux, windows, darwin), `arch` (x86_64, aarch64), `abi` (gnu, msvc,
  darwin), and optionally a minimum glibc (`libc: ">=2.28"`). An artifact is used only on a compatible target.
- `dependencies` are what the artifact needs at link time: system libraries by name, runtimes the integrations know
  (`openmp`), and other artifacts of the package. The compiler reports them; it does not pass linker flags.
- `cubin` and `ptx` (with `device-arch`, e.g. `sm_80`) are reserved: such artifacts are listed as not supported, and
  their implementations are never selected; compilation continues with the others and TVM's code generation.

### Patterns
A pattern is a tree of operator nodes over the Relax graph at the matching boundary (after canonicalization, before
legalization; `mdlc --dump-dir` will write it as `<stem>.relax-match.txt`):
- `{op: <operator>, args: [...], attrs: {...}}`: a call of the operator (e.g. `relax.nn.softmax`, `mdl.matmul`) with
  attributes that must match (e.g. `attrs: { axis: -1 }`);
- `{capture: x}`: any value, named `x` in the implementation. The same name twice must be the same value. A capture
  may be used by other operations outside the region; the operations of the region itself must not be;
- `{const: 2.0}`: a constant, matched exactly; `{const: 0.7978845608, tolerance: 1e-7}` allows a deviation.

`outputs: [a, b]` declares the results of a region with several outputs (by capture name); the default is the root.

### Implementations
- **Signature:** the C function's arguments in order, each with a type (`pointer`, `int64`, `int32`, `float32`,
  `float64`) and a value expression. Pointers declare their access (`read`, `write`, `readwrite`) and optionally an
  alignment. The function returns an `int32`; nonzero is a runtime error that names the kernel. `aliasing` states
  whether the output may share memory with an input. `outputs` give the shape and element type of the results; they
  must agree with the matched region.
- **Applicability:** the element types and layouts per capture, and `where` conditions. A condition that is false or
  cannot be decided at compile time makes the implementation ineligible for that match (`check --explain` says why).
- **Priority:** among overlapping candidates, the higher priority wins, then the larger region, then the package and
  implementation names (deterministic).
- **Adapters:** argument lists that are impractical to write as expressions (e.g. oneDNN's strided matrix products)
  use a named adapter of the compiler instead of `arguments` (`adapter: matmul-strided`).

### Expressions
Values and conditions are written in one small typed language:
- types: `int64`, `bool`, `dtype`, `shape` (a list of `int64`);
- accessors: `$x.shape`, `$x.shape[i]` (negative from the end), `$x.ndim`, `$x.dtype`, `$x.attr.<name>`, `$x.data`
  (pointers only), and `$out` for the result;
- operators: `+ - * // %`, comparisons, `and`, `or`, `not`, `prod(shape[a:b])`, `x in [...]`;
- integer overflow and division by zero are errors; a value that may not fit its argument type (e.g. `int32`) is a
  condition of its own.
