#include <mdl/runtime.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <string_view>

using namespace mdl::runtime;

namespace {
	struct TensorInfo {
		const char* name;
		int32_t ndim;
		std::array<int64_t, 8> shape;
		DLDataType dtype;
	};

	/** @brief Mimics a generated model struct (see the C++ binding generator). */
	struct FakeModel {
		struct Weights {
			const DLTensor* W = nullptr;
			struct {
				const DLTensor* table = nullptr;
			} embed;
			static constexpr std::array<TensorInfo, 2> infos{{
					{"W", 2, {2, 3}, {kDLFloat, 32, 1}},
					{"embed.table", 1, {4}, {kDLInt, 32, 1}},
			}};
			const DLTensor** find(std::string_view path) {
				if (path == "W")
					return &W;
				if (path == "embed.table")
					return &embed.table;
				return nullptr;
			}
		};
	};

	std::filesystem::path tempFile(const char* name) { return std::filesystem::temp_directory_path() / name; }
} // namespace

TEST_CASE("weights are saved and loaded directly into a model's weight slots", "[runtime][safetensors]") {
	Tensor w({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	IntTensor table({4}, std::vector<int32_t>{7, 8, 9, 10});
	FakeModel::Weights weights;
	weights.W = w;
	weights.embed.table = table;
	const auto path = tempFile("mdl_test_weights.safetensors");
	saveWeights<FakeModel>(path, weights);

	auto loaded = loadWeights<FakeModel>(path);
	REQUIRE(loaded.weights.W != nullptr);
	CHECK(static_cast<const float*>(loaded.weights.W->data)[5] == 6.0f);
	CHECK(static_cast<const int32_t*>(loaded.weights.embed.table->data)[3] == 10);
	CHECK(loaded.unused.empty());
	CHECK(reinterpret_cast<uintptr_t>(loaded.weights.W->data) % kTensorAlignment == 0);
	std::filesystem::remove(path);
}

TEST_CASE("loading reports missing, mistyped, and misshaped weights and maps names", "[runtime][safetensors]") {
	Tensor wrongShape({3, 2});
	Tensor extra({1});
	const auto path = tempFile("mdl_test_bad.safetensors");
	saveSafetensors(path, {{"hf.weight", wrongShape}, {"other", extra}});

	// Without mapping: both weights are missing, and both file tensors are unused.
	try {
		loadWeights<FakeModel>(path);
		FAIL("expected an error");
	} catch (const std::runtime_error& e) {
		const std::string message = e.what();
		CHECK(message.find("missing: W") != std::string::npos);
		CHECK(message.find("missing: embed.table") != std::string::npos);
	}
	// With a mapping onto W: the shape mismatch is reported.
	auto map = [](std::string_view name) { return name == "hf.weight" ? std::string{"W"} : std::string{}; };
	try {
		loadWeights<FakeModel>(path, map);
		FAIL("expected an error");
	} catch (const std::runtime_error& e) {
		CHECK(std::string(e.what()).find("W: file has F32[3, 2], model expects F32[2, 3]") != std::string::npos);
	}
	std::filesystem::remove(path);
}
