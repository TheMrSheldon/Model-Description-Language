// Checks an MLP with quantized weight matrices (one build of model.yml): random float weights are written to a
// safetensors file and quantized while loading (mdl::runtime::Quantization::fromFloat); the reference dequantizes the
// loaded parts (W.qweight, W.scales, W.zeros; layout see mdl::builtins::QuantFormat) independently of the runtime and
// computes the MLP in double precision. Also checks that the quantization error of each weight is at most half a step.
// MDL_ELEMENT is the type `float` stands for in the build (float, or bfloat16/float16 with `float: bf16|f16`).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using mdl::runtime::Tensor;
using Model = mdl::models::MLP;
using Element = MDL_ELEMENT;
using ElementTensor = mdl::runtime::BasicTensor<Element>;

/** @brief The loaded float tensor @p name (stored in the map of the model's float type) as doubles. */
template <typename Loaded>
std::vector<double> floatValues(const Loaded& loaded, const std::string& name) {
	auto values = [](const auto& t) { return std::vector<double>(t.values().begin(), t.values().end()); };
	if constexpr (std::is_same_v<Element, float>)
		return values(loaded.floats.at(name));
	else if constexpr (std::is_same_v<Element, mdl::runtime::bfloat16>)
		return values(loaded.bfloats.at(name));
	else
		return values(loaded.halves.at(name));
}

int main() {
	constexpr auto& dims = Model::hyperparameters.dims;
	const int64_t rows = 7;

	// Random float weights for all weights of the float model (W [out, in], b [out]).
	std::mt19937 rng{5};
	std::normal_distribution<float> normal{0.0f, 0.3f};
	std::map<std::string, Tensor> floats;
	for (int layer = 0; layer < 2; ++layer) {
		const std::string prefix = "layers." + std::to_string(layer);
		auto& W = floats.emplace(prefix + ".W", Tensor({int64_t(dims[layer + 1]), int64_t(dims[layer])})).first->second;
		auto& b = floats.emplace(prefix + ".b", Tensor({int64_t(dims[layer + 1])})).first->second;
		for (auto& v : W.values())
			v = normal(rng);
		for (auto& v : b.values())
			v = normal(rng);
	}
	const auto path = std::filesystem::temp_directory_path() / ("mdl_quantization_" + std::to_string(::getpid()) + ".safetensors");
	{
		std::vector<std::pair<std::string, const DLTensor*>> tensors;
		for (const auto& [name, t] : floats)
			tensors.emplace_back(name, t.dl());
		mdl::runtime::saveSafetensors(path, tensors);
	}
	auto loaded = mdl::runtime::loadWeights<Model>(
			path, {}, mdl::runtime::FloatConversion::allowed, mdl::runtime::Quantization::fromFloat
	);
	std::filesystem::remove(path);
	bool ok = loaded.unused.empty();

	// The weights the model computes with: the float weights, or the dequantized parts.
	auto dequantized = [&](const std::string& name, int64_t N, int64_t K, bool& quantized) {
		std::vector<double> W(N * K);
		auto q = loaded.bytes.find(name + ".qweight");
		quantized = q != loaded.bytes.end();
		if (!quantized)
			return floatValues(loaded, name);
		const auto scales = floatValues(loaded, name + ".scales");
		auto zeros = loaded.bytes.find(name + ".zeros");
		const bool int4 = q->second.shape()[1] * 2 == K;
		const int64_t group = K * N / static_cast<int64_t>(scales.size());
		for (int64_t n = 0; n < N; ++n)
			for (int64_t k = 0; k < K; ++k) {
				int value;
				if (!int4) {
					value = q->second.data()[n * K + k];
				} else {
					const auto byte = static_cast<uint8_t>(q->second.data()[n * (K / 2) + k / 2]);
					value = k % 2 == 0 ? byte & 0xF : byte >> 4;
					if (zeros == loaded.bytes.end() && value >= 8)
						value -= 16; // symmetric: signed nibbles
				}
				const int64_t g = k / group;
				if (zeros != loaded.bytes.end())
					value -= zeros->second.data()[g * N + n];
				const double scale = scales[g * N + n];
				W[n * K + k] = value * scale;
				// Round to nearest: at most half a step from the float weight (plus clipping of asymmetric ranges).
				ok &= std::abs(W[n * K + k] - floats.at(name).data()[n * K + k]) <= 0.5 * scale * (1 + 1e-4) + 1e-7;
			}
		return W;
	};

	ElementTensor x({rows, int64_t(dims[0])}), y({rows, int64_t(dims[2])});
	for (auto& v : x.values())
		v = normal(rng);
	Model::forward(x, loaded.weights, y);

	std::vector<double> h(x.values().begin(), x.values().end());
	for (int layer = 0; layer < 2; ++layer) {
		const int64_t K = dims[layer], N = dims[layer + 1];
		const std::string prefix = "layers." + std::to_string(layer);
		bool quantized = false;
		const auto W = dequantized(prefix + ".W", N, K, quantized);
		std::printf("%s.W: %s\n", prefix.c_str(), quantized ? "quantized" : "float");
		const auto b = floatValues(loaded, prefix + ".b");
		std::vector<double> next(rows * N);
		for (int64_t r = 0; r < rows; ++r)
			for (int64_t n = 0; n < N; ++n) {
				double acc = b[n];
				for (int64_t k = 0; k < K; ++k)
					acc += h[r * K + k] * W[n * K + k];
				next[r * N + n] = layer == 0 ? acc * 0.5 * (1 + std::erf(acc / std::sqrt(2.0))) : acc;
			}
		h = std::move(next);
	}
	double maxErr = 0;
	for (size_t i = 0; i < h.size(); ++i)
		maxErr = std::max(maxErr, std::abs(static_cast<double>(y.data()[i]) - h[i]) / std::max(1.0, std::abs(h[i])));
	std::printf("max |mdl - reference| = %.3g (relative to values above 1)\n", maxErr);
	ok &= maxErr < MDL_TOLERANCE;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
