// Calls the compiled FFN through the generated header and compares the result against a reference implementation.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdio>

using mdl::runtime::Tensor;
using Model = mdl::models::FFN;

int main() {
	constexpr int64_t batch = 2, indim = 8, outdim = 3;

	Tensor x({batch, indim}), W({outdim, indim}), b({outdim}), out({batch, outdim});
	for (size_t i = 0; i < x.size(); ++i)
		x.data()[i] = 0.1f * static_cast<float>(i);
	for (size_t i = 0; i < W.size(); ++i)
		W.data()[i] = 0.01f * static_cast<float>(i) - 0.1f;
	for (size_t i = 0; i < b.size(); ++i)
		b.data()[i] = static_cast<float>(i);

	Model::Weights weights;
	weights.W = W;
	weights.b = b;
	Model::forward(x, weights, out);

	int failures = 0;
	for (int64_t r = 0; r < batch; ++r) {
		for (int64_t c = 0; c < outdim; ++c) {
			float expected = b.data()[c];
			for (int64_t k = 0; k < indim; ++k)
				expected += x.data()[r * indim + k] * W.data()[c * indim + k];
			float actual = out.data()[r * outdim + c];
			std::printf("out[%lld, %lld] = %8.4f (expected %8.4f)\n", static_cast<long long>(r),
						static_cast<long long>(c), actual, expected);
			failures += std::abs(actual - expected) > 1e-4f;
		}
	}
	std::printf(failures == 0 ? "OK\n" : "FAILED\n");
	return failures == 0 ? 0 : 1;
}
