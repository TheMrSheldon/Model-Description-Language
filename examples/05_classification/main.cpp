// Checks the objective loss(ids, labels) = cross_entropy(forward(ids), labels) of a gradient build: its value against
// a cross-entropy computed here from forward's logits, and its gradients against central finite differences (including
// the embedding table, whose rows that no id selects must get a zero gradient).
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <tuple>
#include <vector>

using mdl::runtime::IntTensor;
using mdl::runtime::Tensor;
using Model = mdl::models::Classifier;

namespace {
	constexpr int64_t batch = 3;
	constexpr auto conf = Model::hyperparameters;

	std::vector<int64_t> shapeOf(const mdl::models::mdl_detail::TensorInfo& info) {
		std::vector<int64_t> shape(info.shape.begin(), info.shape.begin() + info.ndim);
		for (auto& d : shape)
			if (d == -1)
				d = batch;
		return shape;
	}
} // namespace

int main() {
	std::mt19937 rng{3};
	std::normal_distribution<float> normal{0.0f, 0.5f};
	bool ok = true;

	std::vector<Tensor> tensors;
	tensors.reserve(Model::Weights::infos.size());
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto& t = tensors.emplace_back(shapeOf(info));
		for (auto& v : t.values())
			v = normal(rng);
		*weights.find(info.name) = t;
	}

	// Token ids (with repetitions) and labels.
	IntTensor ids({batch, conf.seq}), labels({batch});
	std::uniform_int_distribution<int32_t> token{0, static_cast<int32_t>(conf.vocab / 2)}; // half the vocabulary unused
	std::uniform_int_distribution<int32_t> label{0, static_cast<int32_t>(conf.classes - 1)};
	for (auto& v : ids.values())
		v = token(rng);
	for (auto& v : labels.values())
		v = label(rng);

	// Reference: the cross-entropy of forward's logits.
	Tensor logits({batch, conf.classes});
	Model::forward(ids, weights, logits);
	double reference = 0;
	for (int64_t b = 0; b < batch; ++b) {
		const float* row = logits.data() + b * conf.classes;
		const double m = *std::max_element(row, row + conf.classes);
		double sum = 0;
		for (int64_t c = 0; c < conf.classes; ++c)
			sum += std::exp(row[c] - m);
		reference += m + std::log(sum) - row[labels.data()[b]];
	}
	reference /= batch;

	// The objective: loss, then the gradients of the weights in canonical order (accumulated: start from zero).
	constexpr size_t numGrads = Model::loss_outputs.size() - 1;
	Tensor loss(std::vector<int64_t>{});
	std::vector<Tensor> grads, scratch;
	for (size_t i = 1; i <= numGrads; ++i) {
		grads.emplace_back(shapeOf(Model::loss_outputs[i]));
		scratch.emplace_back(shapeOf(Model::loss_outputs[i]));
	}
	std::array<DLTensor*, numGrads> gradPtrs, scratchPtrs;
	for (size_t i = 0; i < numGrads; ++i) {
		gradPtrs[i] = grads[i];
		scratchPtrs[i] = scratch[i];
	}
	std::apply([&](auto*... g) { Model::loss(ids, labels, weights, loss, g...); }, gradPtrs);
	std::printf("loss %.6f, reference cross-entropy %.6f\n", loss.data()[0], reference);
	ok &= std::abs(loss.data()[0] - reference) < 1e-5 * std::max(1.0, std::abs(reference));

	auto objective = [&] {
		std::apply([&](auto*... g) { Model::loss(ids, labels, weights, loss, g...); }, scratchPtrs);
		return double(loss.data()[0]);
	};
	double maxRelErr = 0;
	for (size_t p = 0; p < tensors.size(); ++p) {
		Tensor& t = tensors[p];
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
				std::printf("mismatch %s[%zu]: analytic %g, numeric %g\n", Model::loss_outputs[p + 1].name, i, analytic, numeric);
		}
	}
	std::printf("max relative error of the gradients: %.3g\n", maxRelErr);
	ok &= maxRelErr < 1e-2;

	// Rows of the embedding table that no id selects have a zero gradient.
	const std::set<int32_t> used(ids.values().begin(), ids.values().end());
	const Tensor& dTable = grads[0]; // embed.table comes first
	bool unusedZero = true;
	for (int64_t row = 0; row < static_cast<int64_t>(conf.vocab); ++row)
		if (!used.contains(static_cast<int32_t>(row)))
			for (int64_t d = 0; d < static_cast<int64_t>(conf.dim); ++d)
				unusedZero &= dTable.data()[row * conf.dim + d] == 0.0f;
	std::printf("unused embedding rows have zero gradient: %s\n", unusedZero ? "yes" : "no");
	ok &= unusedZero;

	std::puts(ok ? "OK" : "FAILED");
	return ok ? 0 : 1;
}
