// std.var (tests/statistics/model.dlm): the population (dof 0) and sample (dof 1) variance over the last axis of
// x * w, for x [2, 3, 8], against a direct two-pass computation in double precision.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdio>
#include <random>

using mdl::runtime::Tensor;
using Model = mdl::models::Stats;

int main() {
	constexpr int64_t A = 2, B = 3, N = 8;
	std::mt19937 rng{3};
	std::normal_distribution<float> normal{1.0f, 2.0f};
	Tensor x({A, B, N}), w({N});
	for (auto& v : x.values())
		v = normal(rng);
	for (auto& v : w.values())
		v = normal(rng);
	Model::Weights weights;
	*weights.find("w") = w;
	Tensor population({A, B}), sample({A, B});
	Model::population(x, weights, population);
	Model::sample(x, weights, sample);

	double maxErr = 0;
	for (int64_t r = 0; r < A * B; ++r) {
		double mean = 0, ss = 0;
		for (int64_t i = 0; i < N; ++i)
			mean += double(x.data()[r * N + i]) * w.data()[i] / N;
		for (int64_t i = 0; i < N; ++i) {
			const double d = double(x.data()[r * N + i]) * w.data()[i] - mean;
			ss += d * d;
		}
		maxErr = std::max(maxErr, std::abs(population.data()[r] - ss / N) / (ss / N));
		maxErr = std::max(maxErr, std::abs(sample.data()[r] - ss / (N - 1)) / (ss / (N - 1)));
	}
	std::printf("max relative error: %.3g\n", maxErr);
	const bool ok = maxErr < 1e-5;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
