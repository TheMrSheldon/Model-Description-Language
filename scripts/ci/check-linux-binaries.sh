#!/usr/bin/env bash
# Checks that the Linux binaries of an installation run on old distributions: they need at most the given glibc, and
# no libstdc++, libgcc, LLVM, or CUDA library (these are linked statically or loaded at run time).
#   check-linux-binaries.sh <prefix> [max glibc, default 2.28]
set -euo pipefail
prefix=${1:?usage: check-linux-binaries.sh <prefix> [max glibc]}
max=${2:-2.28}
status=0
while IFS= read -r file; do
	# (only ELF files)
	[[ $(head -c 4 "$file" | od -An -c | tr -d ' ') == '177ELF' ]] || continue
	newest=$(objdump -T "$file" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1 || true)
	needed=$(readelf -d "$file" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | tr '\n' ' ')
	echo "${file#"$prefix"/}: glibc ${newest:-none}; needs ${needed}"
	if [[ -n $newest && $(printf '%s\n%s\n' "$newest" "$max" | sort -V | tail -1) != "$max" ]]; then
		echo "  ERROR: needs glibc $newest (> $max)"; status=1
	fi
	# (The CUDA runtime library of the optional cuda component links the CUDA runtime by design: it is only loaded by
	# programs that run models on a GPU.)
	forbidden='libstdc\+\+|libgcc_s|libLLVM|libnvrtc|libcuda|libcudart|libcublas|libcudnn'
	[[ $file == */libtvm_runtime_cuda* ]] && forbidden='libstdc\+\+|libgcc_s|libLLVM'
	if grep -Eq "$forbidden" <<<"$needed"; then
		echo "  ERROR: depends on a library that must not be a dependency"; status=1
	fi
done < <(find "$prefix/bin" "$prefix/lib" -type f \( -perm -u+x -o -name '*.so*' \) 2>/dev/null | sort)
exit $status
