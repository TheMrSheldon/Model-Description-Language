\page installation Installation
\tableofcontents

MDL consists of two parts that can be installed separately:

- the **tooling**: the formatter `mdl-format`, the language server `mdls`, and the standard library. It has no heavy
  dependencies and is what an editor integration needs.
- the **compiler** `mdlc` together with `mdl-kernels`, the runtime library and the kernel packages (e.g. oneDNN). It
  builds on TVM and LLVM.

# Prebuilt tooling
Every release on GitHub (and the rolling `nightly` prerelease) carries archives of the tooling for Linux (x64, arm64),
macOS (universal) and Windows (x64, arm64). Each archive has the layout `bin/mdl-format`, `bin/mdls`,
`share/mdl/std/`; unpack it anywhere and put `bin/` on the `PATH`.

# Building from source
The build uses CMake (at least 3.25 for the presets) and fetches its dependencies with CPM. The grammar is translated by
ANTLR, which needs a Java runtime. The configurations are available as CMake presets:

| Preset | What it builds |
|---|---|
| `conf-tooling` | the frontend, `mdl-format`, `mdls` (no TVM/LLVM/oneDNN) |
| `conf-release-static-deps` | everything, with examples and tests |
| `conf-dev` | everything in debug mode |
| `conf-docs` | this documentation (target `mdl_docs`, see \ref dev_documentation) |

<div class="lang-bash">

```bash
cmake --preset conf-release-static-deps -B build
cmake --build build -j
cmake --install build --prefix ~/.local
```
</div>

The installation contains the standard library (`share/mdl/std`), TVM's libraries and the SDK's headers (`lib/mdl`,
`include/mdl`), the kernel packages (`share/mdl/kernels`) and the CMake package (`lib/cmake/MDL`, see \ref using_cmake).
The installation is relocatable: the commands find their libraries relative to themselves. Building needs CMake 3.28 or
newer.

\todo Describe the required system packages per platform.
