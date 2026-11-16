// Benchmark of the batched encoder (`...` = [batch, seq]) for the benchmark protocol (scripts/benchmark.sh): inference
// (`forward`), or with MDL_BENCH_TRAIN a training step of a gradient build (`forward` saving into the context buffer +
// `forward_backward`). With MDL_BENCH_FLOAT (bfloat16 or float16), the model is compiled with `--float bf16|f16`.
// Prints one line: `<name> median <ms> min <ms> (<iterations>)`, after the warm-up runs.
#include "model.hpp"

#include <mdl/runtime.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#ifdef MDL_BENCH_FLOAT
using Float = mdl::runtime::MDL_BENCH_FLOAT;
#else
using Float = float;
#endif
using Tensor = mdl::runtime::BasicTensor<Float>;
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
} // namespace

int main(int argc, char* argv[]) {
	int64_t batch = 8, seq = 128;
	int warmup = 3, iters = 20;
	std::string name = "bench";
	for (int i = 1; i + 1 < argc; i += 2) {
		const std::string arg = argv[i];
		if (arg == "--batch")
			batch = std::stoll(argv[i + 1]);
		else if (arg == "--seq")
			seq = std::stoll(argv[i + 1]);
		else if (arg == "--warmup")
			warmup = std::stoi(argv[i + 1]);
		else if (arg == "--iters")
			iters = std::stoi(argv[i + 1]);
		else if (arg == "--name")
			name = argv[i + 1];
	}

	std::mt19937 rng{7};
	std::normal_distribution<float> normal{0.0f, 0.02f};
	auto randomize = [&](Tensor& t) {
		for (auto& v : t.values())
			v = Float(normal(rng));
	};
	std::vector<Tensor> tensors;
	tensors.reserve(Model::Weights::infos.size());
	Model::Weights weights;
	for (const auto& info : Model::Weights::infos) {
		auto& t = tensors.emplace_back(shapeOf(info, batch, seq));
		randomize(t);
		*weights.find(info.name) = t;
	}
	Tensor x(shapeOf(Model::forward_inputs[0], batch, seq)), y(shapeOf(Model::forward_outputs[0], batch, seq));
	randomize(x);

#ifdef MDL_BENCH_TRAIN
	constexpr size_t numGrads = Model::forward_backward_outputs.size();
	Tensor dy(y.shape());
	randomize(dy);
	std::vector<Tensor> grads;
	for (const auto& info : Model::forward_backward_outputs)
		grads.emplace_back(shapeOf(info, batch, seq));
	std::array<DLTensor*, numGrads> gradPtrs;
	for (size_t i = 0; i < numGrads; ++i)
		gradPtrs[i] = grads[i];
	auto ctx = Model::forward_ctx_alloc(batch, seq);
	auto step = [&] {
		Model::forward(x, weights, y, ctx);
		std::apply([&](auto*... g) { Model::forward_backward(dy, weights, g..., ctx); }, gradPtrs);
	};
#else
	auto step = [&] { Model::forward(x, weights, y); };
#endif

	for (int i = 0; i < warmup; ++i)
		step();
	std::vector<double> times;
	for (int i = 0; i < iters; ++i) {
		const auto start = std::chrono::steady_clock::now();
		step();
		times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
	}
	std::sort(times.begin(), times.end());
	std::printf("%s median %.4f min %.4f (%d)\n", name.c_str(), times[times.size() / 2], times.front(), iters);
	return 0;
}
