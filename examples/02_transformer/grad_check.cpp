// Checks a gradient build of the batched encoder (`...` = [batch, seq]): forward (which saves the residuals in a context
// buffer) + forward_backward against central finite differences of L = sum(forward(x) * dy). The saved attention
// probabilities grow with seq^2, so this also checks the context layout.
// With MDL_GRAD_MASKED, the encoder is a masked function (`--mask forward`) and the sequences are padded: the gradients
// are those of the masked function, and they do not depend on the padding, not even if it contains NaN.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

using mdl::runtime::Tensor;
using Model = mdl::models::BertEncoder;

namespace {
	std::vector<int64_t> shapeOf(const mdl::models::mdl_detail::TensorInfo& info, int64_t batch, int64_t seq) {
		std::vector<int64_t> shape(info.shape.begin(), info.shape.begin() + info.ndim);
		int unknown = 0;
		for (auto& d : shape)
			if (d == -1)
				d = unknown++ == 0 ? batch : seq;
		return shape;
	}

	constexpr size_t numGrads = Model::forward_backward_outputs.size();
	using GradPtrs = std::array<DLTensor*, numGrads>;

	GradPtrs pointers(std::vector<Tensor>& tensors) {
		GradPtrs ptrs;
		for (size_t i = 0; i < numGrads; ++i)
			ptrs[i] = tensors[i];
		return ptrs;
	}
} // namespace

int main() {
	constexpr int64_t batch = 2, seq = 5;
	std::mt19937 rng{11};
	std::normal_distribution<float> normal{0.0f, 0.3f};
	auto randomize = [&](Tensor& t) {
		for (auto& v : t.values())
			v = normal(rng);
	};
	bool ok = true;
	auto fail = [&](const char* what) {
		std::printf("FAILED: %s\n", what);
		ok = false;
	};

	std::vector<Tensor> tensors;
	tensors.reserve(Model::Weights::infos.size());
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto& t = tensors.emplace_back(shapeOf(info, batch, seq));
		randomize(t);
		*weights.find(info.name) = t;
	}
	Tensor x(shapeOf(Model::forward_inputs[0], batch, seq)), dy(shapeOf(Model::forward_outputs[0], batch, seq));
	randomize(x);
	randomize(dy);
#ifdef MDL_GRAD_MASKED
	// The second sequence has two padding positions.
	mdl::runtime::MaskTensor mask({batch, seq}, uint8_t{1});
	for (int64_t s = seq - 2; s < seq; ++s)
		mask.data()[seq + s] = 0;
	auto forward = [&](const Tensor& input, Tensor& out, Model::Context& c) { Model::forward(input, mask, weights, out, c); };
#else
	auto forward = [&](const Tensor& input, Tensor& out, Model::Context& c) { Model::forward(input, weights, out, c); };
#endif

	// The required size grows with the dimensions; a buffer for larger dimensions is reused.
	const size_t needed = Model::forward_ctx_size(batch, seq);
	std::printf("context buffer: %zu bytes for batch %lld, seq %lld\n", needed, (long long)batch, (long long)seq);
	if (needed == 0 || needed > Model::forward_ctx_size(batch, 2 * seq) || needed > Model::forward_ctx_size(2 * batch, seq))
		fail("context size does not grow with the dimensions");
	auto ctx = Model::forward_ctx_alloc(batch + 1, 2 * seq);

	Tensor y(shapeOf(Model::forward_outputs[0], batch, seq));
	std::vector<Tensor> grads;
	for (const auto& info : Model::forward_backward_outputs) {
		auto& g = grads.emplace_back(shapeOf(info, batch, seq));
		std::fill(g.values().begin(), g.values().end(), 0.0f); // the gradients are accumulated
	}
	auto gradPtrs = pointers(grads);
	forward(x, y, ctx);
	std::apply([&](auto*... g) { Model::forward_backward(dy, weights, g..., ctx); }, gradPtrs);

	// Central finite differences for a few elements of every weight.
	Tensor scratch(shapeOf(Model::forward_outputs[0], batch, seq));
	auto loss = [&] {
		forward(x, scratch, ctx);
		double l = 0;
		for (size_t i = 0; i < scratch.size(); ++i)
			l += double(scratch.data()[i]) * dy.data()[i];
		return l;
	};
	double maxRelErr = 0;
	for (size_t p = 0; p < tensors.size(); ++p) {
		Tensor& t = tensors[p];
		for (size_t i : {size_t{0}, t.size() / 2, t.size() - 1}) {
			const float original = t.data()[i];
			const float eps = 1e-2f * std::max(1.0f, std::abs(original));
			t.data()[i] = original + eps;
			const double plus = loss();
			t.data()[i] = original - eps;
			const double minus = loss();
			t.data()[i] = original;
			const double numeric = (plus - minus) / (2 * eps), analytic = grads[p].data()[i];
			const double relErr = std::abs(numeric - analytic) / std::max(1.0, std::abs(numeric));
			maxRelErr = std::max(maxRelErr, relErr);
			if (relErr > 2e-2)
				std::printf("mismatch %s[%zu]: analytic %g, numeric %g\n", Model::forward_backward_outputs[p].name, i,
							analytic, numeric);
		}
	}
	std::printf("max relative error (finite differences, %zu weights): %.3g\n", tensors.size(), maxRelErr);
	if (maxRelErr > 2e-2)
		fail("gradients differ from finite differences");

#ifdef MDL_GRAD_MASKED
	// NaN in the padding changes neither the output's valid positions nor any gradient.
	Tensor poisoned = x;
	const int64_t hidden = poisoned.shape().back();
	for (int64_t s = seq - 2; s < seq; ++s)
		for (int64_t d = 0; d < hidden; ++d)
			poisoned.data()[(seq + s) * hidden + d] = std::nanf("");
	std::vector<Tensor> again;
	for (const auto& info : Model::forward_backward_outputs) {
		auto& g = again.emplace_back(shapeOf(info, batch, seq));
		std::fill(g.values().begin(), g.values().end(), 0.0f);
	}
	auto againPtrs = pointers(again);
	Tensor y2(y.shape());
	forward(poisoned, y2, ctx);
	std::apply([&](auto*... g) { Model::forward_backward(dy, weights, g..., ctx); }, againPtrs);
	double maxDiff = 0;
	bool finite = true;
	for (size_t g = 0; g < numGrads; ++g)
		for (size_t i = 0; i < grads[g].size(); ++i) {
			finite &= std::isfinite(again[g].data()[i]);
			maxDiff = std::max(maxDiff, double(std::abs(again[g].data()[i] - grads[g].data()[i])));
		}
	for (size_t i = 0; i < y.size(); ++i)
		finite &= std::isfinite(y2.data()[i]);
	std::printf("NaN in the padding: gradients finite: %s, max difference: %.3g\n", finite ? "yes" : "no", maxDiff);
	if (!finite || maxDiff > 1e-5)
		fail("the padding influences the gradients");
#endif

	// A buffer that is too small for the dimensions is rejected.
	auto small = Model::forward_ctx_alloc(batch, seq - 1);
	try {
		forward(x, y, small);
		fail("a too small context buffer was accepted");
	} catch (const std::runtime_error& e) {
		std::printf("too small context rejected: %s\n", e.what());
	}
	Model::forward_ctx_free(small);
	Model::forward_ctx_free(ctx);

	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
