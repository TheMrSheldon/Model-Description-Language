\page dev_packaging Installation and packages
\tableofcontents

The installation rules of the project and the CPack configuration (`cmake/Packaging.cmake`,
`cmake/MDLCPackOptions.cmake`) produce the prebuilt packages of a release. `cmake --install` and `cpack` use the same
components.

# Components
| Component | Contents | In the packages |
|---|---|---|
| `mdl` | the license, the notice and the README | the meta package `mdl` (depends on tools, compiler, sdk) |
| `tools` | `mdl-format`, `mdls`, `mdl-doc`, `share/mdl/std` | `mdl-tools` |
| `runtime` | TVM's runtime libraries (`lib/mdl/libtvm_runtime`, `libtvm_ffi`) | `mdl-runtime` (needed by compiler and sdk) |
| `compiler` | `mdlc`, `mdl-info`, `mdl-kernels`, `lib/mdl/libtvm_compiler`, the CPU kernel packages (oneDNN) | `mdl-compiler` |
| `sdk` | `include/mdl`, the headers of its dependencies (`include/mdl/third_party`), `libmdl_profile`, the CMake package `lib/cmake/MDL` | `mdl-sdk` |
| `cuda` | the CUDA runtime and the kernel packages for CUDA targets (cuBLAS, cuDNN) | `mdl-cuda` (only with `MDL_ENABLE_CUDA`, not on Windows) |

The `sdk` does not depend on the `compiler`: it can be installed alone, for the target of a cross-compilation. The
installation is relocatable: the commands find TVM's libraries through their RPATH (`$ORIGIN/../lib/mdl`), the CMake
package finds everything relative to itself. The dependencies' own install rules must not apply: add each dependency
with `EXCLUDE_FROM_ALL` (the test `install_consumer` fails on files that do not belong to us).

# Formats
| Platform | Formats | Choice of components |
|---|---|---|
| Linux | `.tar.gz` (everything), `.deb`, `.rpm` | one package per component |
| macOS | `.tar.gz` (everything), `.pkg` (productbuild) | the installer's customize page |
| Windows | `.zip` (everything), `.exe` (NSIS) | the component page; install types Full, Compiler, Tools only |

The DEB and RPM packages install to `/usr`, the `.pkg` to `/usr/local`; the `.exe` offers to add `bin` to the `PATH`.
The archives unpack into `mdl-<version>-<platform>/` and work from anywhere. Platforms are `linux-x64`, `linux-arm64`,
`darwin-arm64`, `win32-x64`, `win32-arm64`.

# Building the packages
<div class="lang-bash">

```bash
cmake --preset conf-release-static-deps -B build -G Ninja   # the preset sets CMAKE_INSTALL_LIBDIR=lib
cmake --build build --parallel
ctest --test-dir build -R install_consumer                  # installs, moves the installation, builds against it
cd build && cpack                                           # the generators of the platform (see above)
cpack -G TGZ -D CPACK_MDL_ARCHIVE=tools                     # mdl-tools-<platform>.tar.gz: only the tools (editor integration)
```
</div>

`cpack -G DEB` needs nothing but CPack; `-G RPM` needs `rpmbuild`, `-G NSIS` needs `makensis`.

# Versions
The version is the latest tag `vX.Y.Z`. A prerelease tag (`v1.2.3-rc1`) sorts before the release (DEB `1.2.3~rc1`, RPM
release `0.rc1`), a build after the tag (the nightly, `1.2.3-17-gabc123`) sorts after it (DEB `1.2.3+17.gabc123`).

# Signing
The macOS and Windows packages are not signed. macOS asks to confirm opening the `.pkg` (right-click, Open), and
quarantines downloaded binaries (`xattr -d com.apple.quarantine <file>`, or install through Homebrew, which does not
quarantine); Windows SmartScreen warns on first use. Signing and notarization would need a Developer ID (macOS) and a code
signing certificate (Windows); CPack can sign in a pre-build script (`CPACK_PRE_BUILD_SCRIPTS`, which runs after the
files are staged and stripped).

# License
MDL is under Apache-2.0. Every package carries `LICENSE` and `NOTICE` in `share/doc/mdl-<component>/`, the RPMs state `Apache-2.0`, and the installers show the license on their license page.

# Third-party licenses
The binaries contain third-party software (TVM, LLVM, oneDNN, ANTLR, CLI11, ...) whose licenses require that the license
text and the notices ship with the binaries. Every package contains `share/doc/mdl-<component>/THIRD-PARTY-NOTICES.txt`
with those of the software it contains (one file per component, so that no two packages install the same file). The
places that add a dependency register it with `mdl_third_party()` (`cmake/ThirdParty.cmake`): the install components that
contain it, and its license files. Add a registration with every new dependency that ends up in a binary or in the SDK's
headers. The release builds pass `-DMDL_REQUIRE_THIRD_PARTY_LICENSES=ON`, so that a license that cannot be found (LLVM's:
where the distribution puts it) fails the build instead of only warning.

# The release workflow
`.github/workflows/publish.yml` builds, packages and tries the packages of every platform, then publishes them (tags
`vX.Y.Z`: a release, a pushed commit on `main`: the replaced `nightly` prerelease; a manual run builds and tests only).

1. **Package** (per platform): configure, build, test, and `cpack`. On Linux the build uses the preset
   `conf-release-static-deps` with LLVM linked statically (`-DMDL_LLVM_CONFIG`, an LLVM built by `scripts/ci/build-llvm.sh` and
   cached; it only has the X86 and AArch64 targets), libstdc++ and libgcc linked statically, and CUDA (the toolkit is installed in
   the build container, there is no GPU: the CUDA parts are built, not run). `scripts/ci/check-linux-binaries.sh` checks
   that nothing needs a glibc newer than 2.28 or a libstdc++, LLVM or CUDA library. macOS and Windows build the tooling
   (`conf-tooling`) until the compiler is ported to them.
2. **Smoke tests**: the Linux packages are installed in Debian 11, Ubuntu 20.04, Rocky 8 and Fedora containers
   (`scripts/ci/smoke-test-linux.sh`: the commands run and compile a model), and a program is built against the installed SDK
   on the runner; the macOS `.pkg` and the Windows installer are installed and their tools run.
3. **Publish**: the packages and `SHA256SUMS`. The nightly release is deleted and created again, because the file names
   contain the version.

`test.yml` builds the Linux packages (`.tar.gz`, `.deb`) on every push, so that a change that breaks them fails there.

Changing the LLVM or CUDA version: `LLVM_VERSION` and `CUDA_VERSION` at the top of the workflow (TVM 0.27's CUDA runtime needs
CUDA 12.8 or newer; the LLVM must be one that TVM supports).

# Portability of the Linux packages
The Linux packages are built in a manylinux_2_28 container (glibc 2.28) with libstdc++ and libgcc linked statically, so they
run on any distribution with glibc 2.28 or newer. `mdl-runtime` depends on `libgomp`.
