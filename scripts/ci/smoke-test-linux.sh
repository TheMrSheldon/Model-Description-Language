#!/usr/bin/env bash
# Installs the packages (.deb or .rpm) in a container and runs the commands: the release workflow runs it in old
# distributions to check that the packages install, resolve their dependencies, and work there.
#   smoke-test-linux.sh <packages dir> <example dir>      (the container's package manager decides deb or rpm)
set -euo pipefail
packages=${1:?usage: smoke-test-linux.sh <packages dir> <example dir>}
example=${2:?usage: smoke-test-linux.sh <packages dir> <example dir>}

if command -v apt-get >/dev/null; then
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -qq
	apt-get install -y -qq "$packages"/*.deb
else
	dnf install -y -q "$packages"/*.rpm
fi

mdl-format --version
mdls --version
mdlc --version
mdl-kernels list
# The libraries of the installation resolve.
for file in /usr/bin/mdlc /usr/lib/mdl/libtvm_compiler.so /usr/lib/mdl/libtvm_runtime.so; do
	if ldd "$file" | grep -q 'not found'; then ldd "$file"; echo "ERROR: $file has unresolved libraries"; exit 1; fi
done
# Compile a model (the standard library is found next to the installation).
out=$(mktemp -d)
mdlc "$example/model.dlm" -a "$example/model.yml" -o "$out/model.o" --binding "cpp=$out/model.hpp"
test -s "$out/model.o" && test -s "$out/model.hpp"
echo "smoke test passed"
