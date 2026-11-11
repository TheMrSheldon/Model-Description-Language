// Runs the compiled BERT encoder layer through the generated header, checks it against a straightforward reference
// implementation, and measures its latency.
//
//   02_transformer [--seq N] [--iters N] [--weights file.safetensors]
//
// Without --weights, the weights are random (deterministic). A .safetensors file must contain one float32 tensor per
// weight path of the model (e.g. "weightQuery.W"; see TransformerEncoder::Weights::infos).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using mdl::runtime::Tensor;
using Model = mdl::models::TransformerEncoder;

namespace {
	// ── reference implementation (row-major, naive loops) ────────────────────────────────────────────────────────

	/** y[S, out] = x[S, in] @ W[out, in]^T + b[out] */
	std::vector<float> linear(const std::vector<float>& x, const Tensor& W, const Tensor& b, int64_t S) {
		const int64_t out = W.shape()[0], in = W.shape()[1];
		std::vector<float> y(S * out);
		for (int64_t s = 0; s < S; ++s)
			for (int64_t o = 0; o < out; ++o) {
				double acc = b.data()[o];
				for (int64_t i = 0; i < in; ++i)
					acc += double(x[s * in + i]) * W.data()[o * in + i];
				y[s * out + o] = float(acc);
			}
		return y;
	}

	void layerNorm(std::vector<float>& x, const Tensor& gamma, const Tensor& beta, int64_t S, double eps = 1e-12) {
		const int64_t D = gamma.shape()[0];
		for (int64_t s = 0; s < S; ++s) {
			float* row = x.data() + s * D;
			double mean = 0, var = 0;
			for (int64_t d = 0; d < D; ++d)
				mean += row[d];
			mean /= D;
			for (int64_t d = 0; d < D; ++d)
				var += (row[d] - mean) * (row[d] - mean);
			var /= D;
			for (int64_t d = 0; d < D; ++d)
				row[d] = float((row[d] - mean) / std::sqrt(var + eps) * gamma.data()[d] + beta.data()[d]);
		}
	}

	/** BERT encoder layer (post-LN, exact GELU), heads of size 64. */
	std::vector<float> referenceLayer(const Tensor& xT, std::map<std::string, Tensor>& w, int64_t S) {
		const int64_t D = xT.shape()[1], H = D / 64, Dh = 64;
		std::vector<float> x(xT.values().begin(), xT.values().end());
		auto q = linear(x, w["weightQuery.W"], w["weightQuery.b"], S);
		auto k = linear(x, w["weightKey.W"], w["weightKey.b"], S);
		auto v = linear(x, w["weightValue.W"], w["weightValue.b"], S);
		std::vector<float> ctx(S * D, 0.0f), p(S);
		for (int64_t h = 0; h < H; ++h)
			for (int64_t i = 0; i < S; ++i) {
				double maxScore = -1e300, sum = 0;
				for (int64_t j = 0; j < S; ++j) {
					double score = 0;
					for (int64_t d = 0; d < Dh; ++d)
						score += double(q[i * D + h * Dh + d]) * k[j * D + h * Dh + d];
					p[j] = float(score / std::sqrt(double(Dh)));
					maxScore = std::max(maxScore, double(p[j]));
				}
				for (int64_t j = 0; j < S; ++j)
					sum += (p[j] = float(std::exp(p[j] - maxScore)));
				for (int64_t j = 0; j < S; ++j)
					for (int64_t d = 0; d < Dh; ++d)
						ctx[i * D + h * Dh + d] += float(p[j] / sum * v[j * D + h * Dh + d]);
			}
		auto attn = linear(ctx, w["weightOut.W"], w["weightOut.b"], S);
		for (size_t i = 0; i < attn.size(); ++i)
			attn[i] += x[i];
		layerNorm(attn, w["mhaNorm.gamma"], w["mhaNorm.beta"], S);
		auto inter = linear(attn, w["weightIntermed.W"], w["weightIntermed.b"], S);
		for (auto& value : inter)
			value = float(0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0))));
		auto out = linear(inter, w["weightIntermedOut.W"], w["weightIntermedOut.b"], S);
		for (size_t i = 0; i < out.size(); ++i)
			out[i] += attn[i];
		layerNorm(out, w["ffnNorm.gamma"], w["ffnNorm.beta"], S);
		return out;
	}

	std::vector<int64_t> shapeOf(const mdl::models::mdl_detail::TensorInfo& info) {
		return {info.shape.begin(), info.shape.begin() + info.ndim};
	}
} // namespace

int main(int argc, char* argv[]) {
	int64_t seq = 128;
	int iters = 20;
	std::string weightsFile;
	for (int i = 1; i + 1 < argc; i += 2) {
		if (std::strcmp(argv[i], "--seq") == 0)
			seq = std::stoll(argv[i + 1]);
		else if (std::strcmp(argv[i], "--iters") == 0)
			iters = std::stoi(argv[i + 1]);
		else if (std::strcmp(argv[i], "--weights") == 0)
			weightsFile = argv[i + 1];
	}

	// Weights: loaded from a file or random (LayerNorm scales around 1).
	std::map<std::string, Tensor> tensors;
	if (!weightsFile.empty())
		tensors = mdl::runtime::loadSafetensors(weightsFile);
	std::mt19937 rng{42};
	std::normal_distribution<float> normal{0.0f, 0.02f};
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		if (!tensors.contains(info.name)) {
			Tensor t{shapeOf(info)};
			const bool isScale = std::string_view{info.name}.ends_with("gamma");
			for (auto& value : t.values())
				value = isScale ? 1.0f + 10 * normal(rng) : normal(rng);
			tensors.emplace(info.name, std::move(t));
		}
		*weights.find(info.name) = tensors.at(info.name);
	}

	// Input [seq, model dim] and output.
	auto inShape = shapeOf(Model::forward_inputs[0]);
	auto outShape = shapeOf(Model::forward_outputs[0]);
	std::replace(inShape.begin(), inShape.end(), int64_t{-1}, seq);
	std::replace(outShape.begin(), outShape.end(), int64_t{-1}, seq);
	Tensor x{inShape}, out{outShape};
	std::normal_distribution<float> unit{0.0f, 1.0f};
	for (auto& value : x.values())
		value = unit(rng);

	// Correctness.
	Model::forward(x, weights, out);
	auto expected = referenceLayer(x, tensors, seq);
	double maxErr = 0;
	for (size_t i = 0; i < expected.size(); ++i)
		maxErr = std::max(maxErr, double(std::abs(out.data()[i] - expected[i])));
	std::printf("seq=%lld  max |mdl - reference| = %.3g\n", static_cast<long long>(seq), maxErr);
	if (!(maxErr < 1e-3)) {
		std::printf("FAILED\n");
		return 1;
	}

	// Latency.
	std::vector<double> times;
	for (int i = 0; i < iters; ++i) {
		auto start = std::chrono::steady_clock::now();
		Model::forward(x, weights, out);
		times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
	}
	std::sort(times.begin(), times.end());
	if (!times.empty())
		std::printf("latency: median %.3f ms, min %.3f ms (%d iterations)\n", times[times.size() / 2], times.front(),
					iters);
	std::printf("OK\n");
	return 0;
}
