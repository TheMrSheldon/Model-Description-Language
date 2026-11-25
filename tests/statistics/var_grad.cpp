// The gradient of std.var (tests/statistics/model.dlm, `--grad sample`) w.r.t. the weight w against central finite
// differences of L = sum(sample(x) * dy). The gradient buffer starts non-zero: gradients are added to it.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdio>
#include <random>

using mdl::runtime::Tensor;
using Model = mdl::models::Stats;

int main() {
	constexpr int64_t A = 2, B = 3, N = 8;
	std::mt19937 rng{5};
	std::normal_distribution<float> normal{0.5f, 1.0f};
	Tensor x({A, B, N}), w({N}), dy({A, B});
	for (auto* t : {&x, &w, &dy})
		for (auto& v : t->values())
			v = normal(rng);
	Model::Weights weights;
	*weights.find("w") = w;
	auto ctx = Model::sample_ctx_alloc(A, B);
	Tensor y({A, B}), dw({N});
	constexpr float kPrevious = 0.75f; // (gradients are added to the buffer: checks the update in place)
	std::fill(dw.values().begin(), dw.values().end(), kPrevious);
	Model::sample(x, weights, y, ctx);
	Model::sample_backward(dy, weights, dw, ctx);

	auto loss = [&] {
		Tensor out({A, B});
		auto c = Model::sample_ctx_alloc(A, B);
		Model::sample(x, weights, out, c);
		double l = 0;
		for (int64_t i = 0; i < A * B; ++i)
			l += double(out.data()[i]) * dy.data()[i];
		return l;
	};
	double maxErr = 0;
	for (int64_t i = 0; i < N; ++i) {
		const float original = w.data()[i], eps = 1e-2f;
		w.data()[i] = original + eps;
		const double plus = loss();
		w.data()[i] = original - eps;
		const double minus = loss();
		w.data()[i] = original;
		const double numeric = (plus - minus) / (2 * eps);
		maxErr = std::max(maxErr, std::abs(numeric - (dw.data()[i] - kPrevious)) / std::max(1.0, std::abs(numeric)));
	}
	std::printf("max relative error vs finite differences: %.3g\n", maxErr);
	const bool ok = maxErr < 1e-2;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
