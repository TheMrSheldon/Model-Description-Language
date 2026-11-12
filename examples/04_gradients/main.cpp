// Checks the compiled gradient (forward_vjp) against central finite differences of the compiled forward pass for the
// scalar loss L = sum(forward(x) * dy).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <array>
#include <cmath>
#include <tuple>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using mdl::runtime::Tensor;
using Model = mdl::models::Block;

namespace {
	std::vector<int64_t> shapeOf(const mdl::models::mdl_detail::TensorInfo& info, int64_t rows) {
		std::vector<int64_t> shape(info.shape.begin(), info.shape.begin() + info.ndim);
		for (auto& d : shape)
			if (d == -1)
				d = rows;
		return shape;
	}

	double loss(const Tensor& x, const Model::Weights& weights, const Tensor& dy, Tensor& out) {
		Model::forward(x, weights, out);
		double l = 0;
		for (size_t i = 0; i < out.size(); ++i)
			l += double(out.data()[i]) * dy.data()[i];
		return l;
	}
} // namespace

int main() {
	constexpr int64_t rows = 4;
	std::mt19937 rng{7};
	std::normal_distribution<float> normal{0.0f, 0.5f};
	auto randomize = [&](Tensor& t) {
		for (auto& v : t.values())
			v = normal(rng);
	};

	// Weights, input, and output cotangent.
	std::vector<Tensor> tensors;
	tensors.reserve(Model::Weights::infos.size());
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto& t = tensors.emplace_back(shapeOf(info, rows));
		randomize(t);
		*weights.find(info.name) = t;
	}
	Tensor x(shapeOf(Model::forward_inputs[0], rows)), dy(shapeOf(Model::forward_outputs[0], rows));
	randomize(x);
	randomize(dy);

	// Compiled gradient: outputs are (output, d_<weight>... in canonical order, d_x).
	Tensor y(shapeOf(Model::forward_outputs[0], rows));
	constexpr size_t numGrads = Model::forward_vjp_outputs.size() - 1;
	std::vector<Tensor> grads;
	for (size_t i = 1; i <= numGrads; ++i)
		grads.emplace_back(shapeOf(Model::forward_vjp_outputs[i], rows));
	std::array<DLTensor*, numGrads> gradPtrs;
	for (size_t i = 0; i < numGrads; ++i)
		gradPtrs[i] = grads[i];
	std::apply([&](auto*... g) { Model::forward_vjp(x, dy, weights, y, g...); }, gradPtrs);

	// Finite differences for every weight element and the input.
	std::vector<Tensor*> params;
	for (auto& t : tensors)
		params.push_back(&t);
	params.push_back(&x);
	Tensor scratch(shapeOf(Model::forward_outputs[0], rows));
	double maxRelErr = 0;
	for (size_t p = 0; p < params.size(); ++p) {
		Tensor& t = *params[p];
		for (size_t i = 0; i < t.size(); ++i) {
			const float original = t.data()[i];
			const float eps = 1e-2f * std::max(1.0f, std::abs(original));
			t.data()[i] = original + eps;
			double plus = loss(x, weights, dy, scratch);
			t.data()[i] = original - eps;
			double minus = loss(x, weights, dy, scratch);
			t.data()[i] = original;
			double numeric = (plus - minus) / (2 * eps);
			double analytic = grads[p].data()[i];
			double relErr = std::abs(numeric - analytic) / std::max(1.0, std::abs(numeric));
			maxRelErr = std::max(maxRelErr, relErr);
			if (relErr > 1e-2)
				std::printf("mismatch %s[%zu]: analytic %g, numeric %g\n", Model::forward_vjp_outputs[p + 1].name, i,
							analytic, numeric);
		}
	}
	std::printf("max relative error of %zu gradients: %.3g\n", params.size(), maxRelErr);
	const bool ok = maxRelErr < 1e-2;
	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
