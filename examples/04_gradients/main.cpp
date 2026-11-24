// Checks the compiled gradients of a gradient build against central finite differences.
//  - forward/forward_backward (or, with MDL_FUSED, the fused forward(x, d_output)): the gradient of
//    L = sum(forward(x) * dy) for random dy; the gradients are accumulated (a second call doubles them).
//  - the objective loss(x, target) = mse(forward(x), target): returns the loss and accumulates its gradients.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using mdl::runtime::Tensor;
using Model = mdl::models::Block;

namespace {
	constexpr int64_t rows = 4;

	std::vector<int64_t> shapeOf(const mdl::models::mdl_detail::TensorInfo& info) {
		std::vector<int64_t> shape(info.shape.begin(), info.shape.begin() + info.ndim);
		for (auto& d : shape)
			if (d == -1)
				d = rows;
		return shape;
	}

	/** @brief Gradient buffers for the given output infos (skipping the first `skip` outputs), zeroed. */
	template <size_t N>
	std::vector<Tensor> zeros(const std::array<mdl::models::mdl_detail::TensorInfo, N>& infos, size_t skip) {
		std::vector<Tensor> result;
		for (size_t i = skip; i < N; ++i) {
			auto& t = result.emplace_back(shapeOf(infos[i]));
			std::fill(t.values().begin(), t.values().end(), 0.0f);
		}
		return result;
	}

	template <size_t N>
	std::array<DLTensor*, N> pointers(std::vector<Tensor>& tensors) {
		std::array<DLTensor*, N> ptrs;
		for (size_t i = 0; i < N; ++i)
			ptrs[i] = tensors[i];
		return ptrs;
	}

	/**
	 * @brief Compares @p grads (gradients of the weights in canonical order, then of x) with central finite
	 * differences of @p objective; returns the maximum relative error.
	 */
	double checkFiniteDifferences(
			const char* what, std::vector<Tensor*> params, const std::vector<Tensor>& grads,
			const std::function<double()>& objective
	) {
		double maxRelErr = 0;
		for (size_t p = 0; p < params.size(); ++p) {
			Tensor& t = *params[p];
			for (size_t i = 0; i < t.size(); ++i) {
				const float original = t.data()[i];
				const float eps = 1e-2f * std::max(1.0f, std::abs(original));
				t.data()[i] = original + eps;
				const double plus = objective();
				t.data()[i] = original - eps;
				const double minus = objective();
				t.data()[i] = original;
				const double numeric = (plus - minus) / (2 * eps), analytic = grads[p].data()[i];
				const double relErr = std::abs(numeric - analytic) / std::max(1.0, std::abs(numeric));
				maxRelErr = std::max(maxRelErr, relErr);
				if (relErr > 1e-2)
					std::printf("%s: mismatch of gradient %zu [%zu]: analytic %g, numeric %g\n", what, p, i, analytic, numeric);
			}
		}
		std::printf("%s: max relative error of %zu gradients: %.3g\n", what, params.size(), maxRelErr);
		return maxRelErr;
	}
} // namespace

int main() {
	std::mt19937 rng{7};
	std::normal_distribution<float> normal{0.0f, 0.5f};
	auto randomize = [&](Tensor& t) {
		for (auto& v : t.values())
			v = normal(rng);
	};
	bool ok = true;

	std::vector<Tensor> tensors;
	tensors.reserve(Model::Weights::infos.size());
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto& t = tensors.emplace_back(shapeOf(info));
		randomize(t);
		*weights.find(info.name) = t;
	}
	Tensor x(shapeOf(Model::forward_inputs[0])), y(shapeOf(Model::forward_outputs[0])), dy(y.shape());
	randomize(x);
	randomize(dy);
	std::vector<Tensor*> params;
	for (auto& t : tensors)
		params.push_back(&t);
	params.push_back(&x);

#ifdef MDL_FUSED
	// forward(x, d_output) -> output, d_<weight>..., d_x
	constexpr size_t numGrads = Model::forward_outputs.size() - 1;
	auto grads = zeros(Model::forward_outputs, 1);
	auto gradPtrs = pointers<numGrads>(grads);
	auto run = [&](std::vector<Tensor>& g) {
		auto ptrs = pointers<numGrads>(g);
		std::apply([&](auto*... p) { Model::forward(x, dy, weights, y, p...); }, ptrs);
	};
	run(grads);
	auto scratch = zeros(Model::forward_outputs, 1);
	auto objective = [&] {
		run(scratch);
		double l = 0;
		for (size_t i = 0; i < y.size(); ++i)
			l += double(y.data()[i]) * dy.data()[i];
		return l;
	};
	(void)gradPtrs;
#else
	// forward(x) saves the residuals; forward_backward(d_output) -> d_<weight>..., d_x. The context buffer is allocated
	// for a larger batch than used (a buffer for the largest dimensions is reused).
	// [backward]
	constexpr size_t numGrads = Model::forward_backward_outputs.size();
	auto ctx = Model::forward_ctx_alloc(4 * rows);
	auto grads = zeros(Model::forward_backward_outputs, 0);
	auto gradPtrs = pointers<numGrads>(grads);
	Model::forward(x, weights, y, ctx);
	std::apply([&](auto*... g) { Model::forward_backward(dy, weights, g..., ctx); }, gradPtrs);
	// [backward]
	auto objective = [&] {
		Model::forward(x, weights, y, ctx);
		double l = 0;
		for (size_t i = 0; i < y.size(); ++i)
			l += double(y.data()[i]) * dy.data()[i];
		return l;
	};
#endif
	ok &= checkFiniteDifferences("forward", params, grads, objective) < 1e-2;

#ifndef MDL_FUSED
	// Accumulation: a second backward pass adds the same gradients again.
	std::vector<std::vector<float>> once;
	for (auto& g : grads)
		once.emplace_back(g.values().begin(), g.values().end());
	Model::forward(x, weights, y, ctx);
	std::apply([&](auto*... g) { Model::forward_backward(dy, weights, g..., ctx); }, gradPtrs);
	double maxDiff = 0;
	for (size_t g = 0; g < grads.size(); ++g)
		for (size_t i = 0; i < grads[g].size(); ++i)
			maxDiff = std::max(maxDiff, double(std::abs(grads[g].data()[i] - 2 * once[g][i])));
	std::printf("accumulation: max difference from twice the gradient: %.3g\n", maxDiff);
	ok &= maxDiff < 1e-5;
	Model::forward_ctx_free(ctx);

	// The objective: loss(x, target) -> loss, d_<weight>..., d_x (accumulated).
	constexpr size_t numLossGrads = Model::loss_outputs.size() - 1;
	Tensor target(y.shape()), loss(std::vector<int64_t>{});
	randomize(target);
	auto lossGrads = zeros(Model::loss_outputs, 1);
	auto lossScratch = zeros(Model::loss_outputs, 1);
	auto lossPtrs = pointers<numLossGrads>(lossGrads);
	auto scratchPtrs = pointers<numLossGrads>(lossScratch);
	// [loss]
	std::apply([&](auto*... g) { Model::loss(x, target, weights, loss, g...); }, lossPtrs);
	// [loss]
	auto lossObjective = [&] {
		std::apply([&](auto*... g) { Model::loss(x, target, weights, loss, g...); }, scratchPtrs);
		return double(loss.data()[0]);
	};
	ok &= checkFiniteDifferences("loss", params, lossGrads, lossObjective) < 1e-2;
#endif

	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
