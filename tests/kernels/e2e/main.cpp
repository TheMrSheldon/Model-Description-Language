// End to end: a model compiled with the kernel package scaled_add_c calls its kernel for `2 * x + y` (forward) and
// TVM-generated code elsewhere; `3 * x + y` (other, inner) has a kernel with a runtime guard (even batch sizes), so
// both branches run. All give the reference results.
#include "scaled_add.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

extern "C" int64_t test_scaled_add_calls, test_scaled3_add_calls; // (counted by the kernels)

using mdl::runtime::Tensor;
using Model = mdl::models::M;

int main() {
	constexpr int64_t n = 8;
	bool ok = true;
	for (int64_t batch : {3, 4}) {
		Tensor x({batch, n}), y({batch, n}), out({batch, n});
		for (int64_t i = 0; i < batch * n; ++i) {
			x.data()[i] = 0.25f * static_cast<float>(i);
			y.data()[i] = 1.0f - static_cast<float>(i % 5);
		}
		auto check = [&](const char* name, float scale, float factor, int64_t calls, int64_t calls3) {
			float maxErr = 0;
			for (int64_t i = 0; i < batch * n; ++i)
				maxErr = std::fmax(maxErr, std::fabs(out.data()[i] - factor * (scale * x.data()[i] + y.data()[i])));
			std::printf(
					"batch %lld, %s: max error %g, kernel calls %lld, %lld\n", static_cast<long long>(batch), name,
					maxErr, static_cast<long long>(test_scaled_add_calls),
					static_cast<long long>(test_scaled3_add_calls)
			);
			ok &= maxErr == 0.0f && test_scaled_add_calls == calls && test_scaled3_add_calls == calls3;
		};
		test_scaled_add_calls = test_scaled3_add_calls = 0;
		const int64_t guarded = batch % 2 == 0 ? 1 : 0; // the guard holds for even batch sizes
		Model::forward(x, y, {}, out);
		check("forward (kernel package)", 2.0f, 1.0f, 1, 0);
		Model::other(x, y, {}, out);
		check("other (guarded: kernel or TVM)", 3.0f, 1.0f, 1, guarded);
		Model::inner(x, y, {}, out);
		check("inner (guarded, not the result)", 3.0f, 0.5f, 1, 2 * guarded);
	}
	std::printf(ok ? "OK\n" : "FAILED\n");
	return ok ? 0 : 1;
}
