// Checks the MLP of model.dlm (array hyperparameters from model.yml, a layer without bias, a default) against a
// reference implementation, and that the header exposes the hyperparameters (arrays as std::array).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using mdl::runtime::Tensor;
using Model = mdl::models::MLP;

int main() {
	constexpr auto& hp = Model::hyperparameters;
	static_assert(hp.conf.sizes.size() == 4 && hp.conf.sizes[1] == 16 && hp.conf.biases[2] == 0 && hp.normalize == 0);
	const int64_t rows = 5, in = hp.conf.sizes[0], out = hp.conf.sizes[3];

	// The last layer has no bias: no weight layers.2.b.
	bool ok = true;
	for (const auto& info : Model::Weights::infos)
		ok &= std::string(info.name) != "layers.2.b";
	std::printf("weights: %zu (layers.2 without bias)\n", Model::Weights::infos.size());

	std::mt19937 rng{3};
	std::normal_distribution<float> normal{0.0f, 0.5f};
	std::vector<Tensor> tensors;
	tensors.reserve(Model::Weights::infos.size());
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto& t = tensors.emplace_back(std::vector<int64_t>(info.shape.begin(), info.shape.begin() + info.ndim));
		for (auto& v : t.values())
			v = normal(rng);
		*weights.find(info.name) = t;
	}
	Tensor x({rows, in}), y({rows, out});
	for (auto& v : x.values())
		v = normal(rng);
	Model::forward(x, weights, y);

	// Reference: h = gelu(h W0^T + b0), h = gelu(h W1^T + b1), y = h W2^T.
	auto weight = [&](const std::string& name) -> const Tensor* {
		for (size_t i = 0; i < Model::Weights::infos.size(); ++i)
			if (Model::Weights::infos[i].name == name)
				return &tensors[i];
		return nullptr;
	};
	std::vector<float> h(x.values().begin(), x.values().end());
	for (int layer = 0; layer < 3; ++layer) {
		const int64_t n = hp.conf.sizes[layer], m = hp.conf.sizes[layer + 1];
		const Tensor* W = weight("layers." + std::to_string(layer) + ".W");
		const Tensor* b = weight("layers." + std::to_string(layer) + ".b");
		std::vector<float> next(rows * m);
		for (int64_t r = 0; r < rows; ++r)
			for (int64_t j = 0; j < m; ++j) {
				double acc = b ? b->data()[j] : 0.0;
				for (int64_t k = 0; k < n; ++k)
					acc += h[r * n + k] * W->data()[j * n + k];
				next[r * m + j] = layer < 2 ? static_cast<float>(acc * 0.5 * (1 + std::erf(acc / std::sqrt(2.0)))) : static_cast<float>(acc);
			}
		h = std::move(next);
	}
	double maxErr = 0;
	for (size_t i = 0; i < h.size(); ++i)
		maxErr = std::max(maxErr, double(std::abs(y.data()[i] - h[i])));
	std::printf("max |mdl - reference| = %.3g\n", maxErr);
	ok &= maxErr < 1e-4;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
