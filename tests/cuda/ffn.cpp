// The FFN of examples/01_ffn compiled with --target cuda: called with device tensors through the generated header, and
// with a tensor on the wrong device (a clean error naming it, not a crash).
#include "model.hpp"

#include <tvm/runtime/tensor.h>

#include <cmath>
#include <cstdio>
#include <vector>

using tvm::runtime::Tensor;

int main() {
	constexpr int64_t batch = 2, indim = 8, outdim = 3;
	DLDevice gpu{kDLCUDA, 0}, cpu{kDLCPU, 0};
	DLDataType f32{kDLFloat, 32, 1};
	std::vector<float> x(batch * indim), W(outdim * indim), b(outdim), out(batch * outdim);
	for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * i;
	for (size_t i = 0; i < W.size(); ++i) W[i] = 0.01f * i - 0.1f;
	for (size_t i = 0; i < b.size(); ++i) b[i] = i;
	auto dx = Tensor::Empty({batch, indim}, f32, gpu), dW = Tensor::Empty({outdim, indim}, f32, gpu),
		 db = Tensor::Empty({outdim}, f32, gpu), dout = Tensor::Empty({batch, outdim}, f32, gpu);
	dx.CopyFromBytes(x.data(), x.size() * 4);
	dW.CopyFromBytes(W.data(), W.size() * 4);
	db.CopyFromBytes(b.data(), b.size() * 4);

	mdl::models::FFN::Weights weights;
	weights.W = dW.operator->();
	weights.b = db.operator->();
	mdl::models::FFN::forward(dx.operator->(), weights, const_cast<DLTensor*>(dout.operator->()));
	dout.CopyToBytes(out.data(), out.size() * 4);

	int failures = 0;
	for (int64_t r = 0; r < batch; ++r)
		for (int64_t c = 0; c < outdim; ++c) {
			float expected = b[c];
			for (int64_t k = 0; k < indim; ++k) expected += x[r * indim + k] * W[c * indim + k];
			std::printf("out[%ld,%ld] = %8.4f (expected %8.4f)\n", r, c, out[r * outdim + c], expected);
			failures += std::abs(out[r * outdim + c] - expected) > 1e-4f;
		}

	auto hx = Tensor::Empty({batch, indim}, f32, cpu);
	try {
		mdl::models::FFN::forward(hx.operator->(), weights, const_cast<DLTensor*>(dout.operator->()));
		std::printf("wrong device: no error\n");
		++failures;
	} catch (const std::exception& e) {
		std::printf("wrong device: %s\n", e.what());
	}
	std::printf(failures == 0 ? "OK\n" : "FAILED\n");
	return failures == 0 ? 0 : 1;
}
