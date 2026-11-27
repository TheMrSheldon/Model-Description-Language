#!/usr/bin/env bash
# Builds a static LLVM (only what TVM's code generation needs: the X86 and AArch64 targets and the libraries) for the Linux
# packages, which link it into the compiler library so that they need no LLVM on the system. The release workflow caches
# the result.
#   build-llvm.sh <prefix> <version>        e.g. build-llvm.sh /opt/llvm 21.1.0
set -euo pipefail
prefix=${1:?usage: build-llvm.sh <prefix> <version>}
version=${2:?usage: build-llvm.sh <prefix> <version>}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone --quiet --depth 1 --branch "llvmorg-${version}" https://github.com/llvm/llvm-project "$work/llvm-project"
cmake -S "$work/llvm-project/llvm" -B "$work/build" -G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="$prefix" \
	-DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
	-DLLVM_ENABLE_ASSERTIONS=OFF \
	-DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF \
	-DLLVM_ENABLE_Z3_SOLVER=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
	-DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF \
	-DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$work/build" --parallel
cmake --install "$work/build" --strip
# Only llvm-config is needed of the tools (TVM asks it for the libraries): the others are 1 GB that the cache need not keep.
find "$prefix/bin" -mindepth 1 ! -name llvm-config -delete
# The license goes with the binaries that contain LLVM (cmake/ThirdParty.cmake looks for it next to llvm-config).
mkdir -p "$prefix/share/doc/llvm"
cp "$work/llvm-project/llvm/LICENSE.TXT" "$prefix/share/doc/llvm/LICENSE.TXT"
"$prefix/bin/llvm-config" --version
