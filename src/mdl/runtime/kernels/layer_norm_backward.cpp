// The gradient of a layer normalization over the last axis with respect to its input, in one pass per row (the
// kernel package onednn, pattern layer-norm-backward):
//   dx = rsqrt(var) * (g - mean(g) - xhat * mean(g * xhat)),  g = dy * gamma
// where xhat is the normalized input and var the variance plus epsilon (both saved by the forward pass). 16-bit types
// are computed in float32.
#include <mdl/runtime/kernels.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

	struct F32 {
		using type = float;
		static float load(float v) { return v; }
		static float store(float v) { return v; }
	};

	struct BF16 {
		using type = uint16_t;
		static float load(uint16_t v) {
			const uint32_t bits = static_cast<uint32_t>(v) << 16;
			float f;
			std::memcpy(&f, &bits, sizeof(f));
			return f;
		}
		static uint16_t store(float f) {
			uint32_t bits;
			std::memcpy(&bits, &f, sizeof(bits));
			if (std::isnan(f))
				return static_cast<uint16_t>((bits >> 16) | 0x40);
			bits += 0x7fff + ((bits >> 16) & 1); // round to nearest even
			return static_cast<uint16_t>(bits >> 16);
		}
	};

	struct F16 {
		using type = uint16_t;
		static float load(uint16_t v) {
			_Float16 h;
			std::memcpy(&h, &v, sizeof(h));
			return static_cast<float>(h);
		}
		static uint16_t store(float f) {
			const auto h = static_cast<_Float16>(f);
			uint16_t v;
			std::memcpy(&v, &h, sizeof(v));
			return v;
		}
	};

	template <typename T>
	int layerNormBackward(
			typename T::type* dx, const typename T::type* dy, const typename T::type* gamma,
			const typename T::type* xhat, const typename T::type* var, int64_t rows, int64_t channels
	) {
		if (rows < 0 || channels <= 0)
			return 1;
		std::vector<float> scale(static_cast<size_t>(channels));
		for (int64_t c = 0; c < channels; ++c)
			scale[c] = T::load(gamma[c]);
		const float inv = 1.0f / static_cast<float>(channels);
#pragma omp parallel for schedule(static)
		for (int64_t r = 0; r < rows; ++r) {
			const auto* dyRow = dy + r * channels;
			const auto* xRow = xhat + r * channels;
			auto* dxRow = dx + r * channels;
			float sumG = 0, sumGX = 0;
#pragma omp simd reduction(+ : sumG, sumGX)
			for (int64_t c = 0; c < channels; ++c) {
				const float g = T::load(dyRow[c]) * scale[c];
				sumG += g;
				sumGX += g * T::load(xRow[c]);
			}
			const float meanG = sumG * inv, meanGX = sumGX * inv;
			const float rstd = 1.0f / std::sqrt(T::load(var[r]));
#pragma omp simd
			for (int64_t c = 0; c < channels; ++c) {
				const float g = T::load(dyRow[c]) * scale[c];
				dxRow[c] = T::store(rstd * (g - meanG - T::load(xRow[c]) * meanGX));
			}
		}
		return 0;
	}

} // namespace

#define MDL_LAYER_NORM_BACKWARD(suffix, Traits)                                                                        \
	extern "C" int mdl_layer_norm_backward_##suffix(                                                                   \
			Traits::type* dx, const Traits::type* dy, const Traits::type* gamma, const Traits::type* xhat,             \
			const Traits::type* var, int64_t rows, int64_t channels                                                    \
	) {                                                                                                                \
		return layerNormBackward<Traits>(dx, dy, gamma, xhat, var, rows, channels);                                    \
	}
MDL_LAYER_NORM_BACKWARD(f32, F32)
MDL_LAYER_NORM_BACKWARD(bf16, BF16)
MDL_LAYER_NORM_BACKWARD(f16, F16)
#undef MDL_LAYER_NORM_BACKWARD
