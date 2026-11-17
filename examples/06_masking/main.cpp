// Averages a masked sequence with the masked overload (`sum(x.data) / sum(x.mask)`) and with generic masking (`mean`),
// and compares both with the average of the valid elements computed here; likewise the mean and sum of all valid
// elements (mean_flat, sum_flat).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdio>

using Model = mdl::models::Pool;

int main() {
	constexpr int64_t seq = 5, dim = Model::hyperparameters.dim;
	mdl::runtime::Tensor x({seq, dim});
	for (int64_t s = 0; s < seq; ++s)
		for (int64_t d = 0; d < dim; ++d)
			x.data()[s * dim + d] = static_cast<float>(10 * s + d);
	mdl::runtime::MaskTensor mask({seq}, std::vector<uint8_t>{1, 0, 1, 1, 0}); // positions 0, 2, 3 are valid
	x.data()[1 * dim] = std::nanf("");										   // masked values do not matter

	mdl::runtime::Tensor overloadOut({dim}), genericOut({dim});
	Model::Weights weights;
	Model::withOverload(x, mask, weights, overloadOut);
	Model::generic(x, mask, weights, genericOut);

	bool ok = true;
	for (int64_t d = 0; d < dim; ++d) {
		const float expected = (0.0f + 20.0f + 30.0f) / 3 + d;
		std::printf(
				"d=%lld: overload %g, generic %g, expected %g\n", (long long)d, overloadOut.data()[d],
				genericOut.data()[d], expected
		);
		ok &= std::abs(overloadOut.data()[d] - expected) < 1e-5 && std::abs(genericOut.data()[d] - expected) < 1e-5;
	}
	// All valid elements (rows 0, 2, 3): their mean and sum.
	mdl::runtime::Tensor meanOut(std::vector<int64_t>{}), sumOut(std::vector<int64_t>{});
	Model::meanAll(x, mask, weights, meanOut);
	Model::sumAll(x, mask, weights, sumOut);
	float expectedSum = 0;
	for (int64_t s : {0, 2, 3})
		for (int64_t d = 0; d < dim; ++d)
			expectedSum += static_cast<float>(10 * s + d);
	const float expectedMean = expectedSum / static_cast<float>(3 * dim);
	std::printf(
			"all: mean %g (expected %g), sum %g (expected %g)\n", meanOut.data()[0], expectedMean, sumOut.data()[0],
			expectedSum
	);
	ok &= std::abs(meanOut.data()[0] - expectedMean) < 1e-4 && std::abs(sumOut.data()[0] - expectedSum) < 1e-3;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
