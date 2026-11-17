// The kernel of the end-to-end test package: out = 2 * x + y over n elements; counts its calls.
#include <stdint.h>

int64_t test_scaled_add_calls = 0;

int test_scaled_add_f32(float* out, const float* x, const float* y, int64_t n) {
	for (int64_t i = 0; i < n; ++i)
		out[i] = 2.0f * x[i] + y[i];
	++test_scaled_add_calls;
	return 0;
}
