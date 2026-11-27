\page installation Installation
\tableofcontents

MDL consists of two parts that can be installed separately:

- the **tooling**: the formatter `mdl-format`, the language server `mdls`, and the standard library. It has no heavy
  dependencies and is what an editor integration needs.
- the **compiler** `mdlc` together with `mdl-kernels`, the runtime library and the kernel packages (e.g. oneDNN). It
  builds on TVM and LLVM.

# Prebuilt packages
Every release on GitHub (and the rolling `nightly` prerelease) carries installers and archives. `SHA256SUMS` lists their
checksums.

| Platform | Contents | Installers | Archive |
|---|---|---|---|
| Linux x64, arm64 | everything (compiler, runtime, SDK, tools; CUDA) | `.deb` and `.rpm`, one package per component | `mdl-<version>-linux-<arch>.tar.gz` |
| macOS arm64 | the tools | `.pkg` | `mdl-<version>-darwin-arm64.tar.gz` |
| Windows x64, arm64 | the tools | `.exe` installer | `mdl-<version>-win32-<arch>.zip` |

The compiler is not yet available prebuilt on macOS and Windows; build it from source (below).

## Linux
The packages run on distributions with glibc 2.28 or newer (Debian 10, Ubuntu 18.04, RHEL 8, and later). Install all of
them, or only what you need:

<div class="lang-bash">

```bash
sudo apt install ./mdl_*.deb ./mdl-*.deb     # or: sudo dnf install ./*.rpm
# only the tools: sudo apt install ./mdl-tools_*.deb
```
</div>

| Package | Contents |
|---|---|
| `mdl` | meta package: depends on `mdl-tools`, `mdl-compiler` and `mdl-sdk` |
| `mdl-tools` | `mdl-format`, `mdls`, `mdl-doc`, and the standard library |
| `mdl-compiler` | `mdlc`, `mdl-info`, `mdl-kernels` and the CPU kernel libraries (needs `mdl-tools`, `mdl-runtime`) |
| `mdl-sdk` | headers, libraries and the CMake package to use compiled models in C++ programs (needs `mdl-runtime`) |
| `mdl-runtime` | TVM's runtime libraries |
| `mdl-cuda` | compile for NVIDIA GPUs (needs the CUDA toolkit on the machine; not installed by `mdl`) |

## macOS and Windows
Run the installer: the Windows installer lets you choose the components and add MDL to the `PATH`. The installers are
not signed: macOS asks you to confirm opening the `.pkg` (right-click, Open), and Windows SmartScreen warns on first use.
The archives unpack anywhere; put `bin/` on the `PATH`. For the editor integration, `mdl-tools-<platform>.{tar.gz,zip}`
contains only the tools.

# Building from source
The build uses CMake (at least 3.28) and fetches its dependencies with CPM. The grammar is translated by
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
