// End to end: a model compiled with the kernel package scaled_add_c calls its kernel for `2 * x + y` (forward) and
// TVM-generated code elsewhere (other: 3 * x + y); both give the reference results.
#include "scaled_add.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

extern "C" int64_t test_scaled_add_calls; // (counted by the kernel)

using mdl::runtime::Tensor;
using Model = mdl::models::M;

int main() {
	constexpr int64_t batch = 3, n = 8;
	Tensor x({batch, n}), y({batch, n}), out({batch, n});
	for (int64_t i = 0; i < batch * n; ++i) {
		x.data()[i] = 0.25f * static_cast<float>(i);
		y.data()[i] = 1.0f - static_cast<float>(i % 5);
	}
	bool ok = true;
	auto check = [&](const char* name, float scale, int64_t expectedCalls) {
		float maxErr = 0;
		for (int64_t i = 0; i < batch * n; ++i)
			maxErr = std::fmax(maxErr, std::fabs(out.data()[i] - (scale * x.data()[i] + y.data()[i])));
		std::printf(
				"%s: max error %g, kernel calls %lld\n", name, maxErr, static_cast<long long>(test_scaled_add_calls)
		);
		ok &= maxErr == 0.0f && test_scaled_add_calls == expectedCalls;
	};
	Model::forward(x, y, {}, out);
	check("forward (kernel package)", 2.0f, 1);
	Model::other(x, y, {}, out);
	check("other (TVM)", 3.0f, 1);
	std::printf(ok ? "OK\n" : "FAILED\n");
	return ok ? 0 : 1;
}
