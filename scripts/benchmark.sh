#!/usr/bin/env bash
# The benchmark protocol: BERT-base and tiny encoder, inference and training step, float32 and bfloat16, batch 8 x
# sequence 128 (tiny: batch 8 x 32), a fixed number of threads, 3 warm-up and 20 timed runs; prints the medians.
# Usage: scripts/benchmark.sh BUILD_DIR [OUTPUT_FILE]
#   BUILD_DIR: a Release build configured with -DMDL_BUILD_EXAMPLES=ON -DMDL_BENCHMARKS=ON (built here).
#   Environment: OMP_NUM_THREADS (default: 8) and TVM_NUM_THREADS (default: the same).
# Compare two runs with: paste OLD NEW, or scripts/benchmark.sh BUILD_DIR new.txt and diff against old.txt.
set -euo pipefail
build=${1:?usage: scripts/benchmark.sh BUILD_DIR [OUTPUT_FILE]}
output=${2:-/dev/stdout}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TVM_NUM_THREADS=${TVM_NUM_THREADS:-$OMP_NUM_THREADS}

targets=()
for size in base tiny; do
	for mode in infer train; do
		for precision in f32 bf16; do
			targets+=("bench_${size}_${mode}_${precision}")
		done
	done
done
cmake --build "$build" --target "${targets[@]}" --parallel >&2

{
	echo "# $(date -u +%FT%TZ) $(git -C "$(dirname "$0")/.." describe --always --dirty) threads=$OMP_NUM_THREADS"
	for target in "${targets[@]}"; do
		seq=128
		[[ $target == bench_tiny_* ]] && seq=32
		"$build/examples/02_transformer/$target" --name "$target" --batch 8 --seq "$seq" --warmup 3 --iters 20
	done
} | tee "$output"
