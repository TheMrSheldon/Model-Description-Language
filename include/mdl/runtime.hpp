#ifndef MDL_RUNTIME_HPP
#define MDL_RUNTIME_HPP

/**
 * @file
 * @brief Small header-only helpers for programs that call compiled models: owning (CPU; float32, bfloat16, float16,
 * int32, bool) tensors that can be passed as `DLTensor*`, and safetensors support: loading a model's weights directly into its generated `Weights`
 * (with dtype and shape checks and an optional name mapping), saving them, and reading arbitrary tensors.
 */

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>

#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <vector>

namespace mdl::runtime {

	/** @brief Alignment (in bytes) that compiled functions require for tensor data. */
	inline constexpr size_t kTensorAlignment = 64;

	/** @brief Allocator for tensor data aligned to kTensorAlignment. */
	template <typename T>
	struct AlignedAllocator {
		using value_type = T;
		AlignedAllocator() = default;
		template <typename U>
		AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
		T* allocate(size_t n) {
			return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{kTensorAlignment}));
		}
		void deallocate(T* p, size_t) noexcept { ::operator delete(p, std::align_val_t{kTensorAlignment}); }
		template <typename U>
		bool operator==(const AlignedAllocator<U>&) const noexcept {
			return true;
		}
	};

	/** @brief A bfloat16 value (float32's exponent with 8 bits of precision): the upper half of a float32. */
	struct bfloat16 {
		uint16_t bits = 0;

		bfloat16() = default;
		bfloat16(float value) { // rounds to nearest even
			uint32_t x;
			std::memcpy(&x, &value, sizeof(x));
			if ((x & 0x7fffffffu) > 0x7f800000u) // NaN stays NaN
				bits = static_cast<uint16_t>((x >> 16) | 0x40u);
			else
				bits = static_cast<uint16_t>((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
		}
		operator float() const {
			const uint32_t x = static_cast<uint32_t>(bits) << 16;
			float value;
			std::memcpy(&value, &x, sizeof(value));
			return value;
		}
	};

	/** @brief An IEEE 754 half-precision value. */
	struct float16 {
		uint16_t bits = 0;

		float16() = default;
		float16(float value) { // rounds to nearest even
			uint32_t x;
			std::memcpy(&x, &value, sizeof(x));
			const uint32_t sign = (x >> 16) & 0x8000u;
			const uint32_t absolute = x & 0x7fffffffu;
			if (absolute >= 0x7f800000u) { // inf or NaN
				bits = static_cast<uint16_t>(sign | 0x7c00u | (absolute > 0x7f800000u ? 0x200u : 0u));
			} else if (absolute >= 0x477ff000u) { // overflows to inf
				bits = static_cast<uint16_t>(sign | 0x7c00u);
			} else if (absolute < 0x38800000u) { // subnormal (or zero): round the scaled value
				float scaled;
				const uint32_t a = absolute;
				std::memcpy(&scaled, &a, sizeof(scaled));
				bits = static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(scaled * 16777216.0f)));
			} else {
				const uint32_t rounded = absolute + 0xfffu + ((absolute >> 13) & 1u);
				bits = static_cast<uint16_t>(sign | ((rounded - 0x38000000u) >> 13));
			}
		}
		operator float() const {
			const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
			const uint32_t exponent = (bits >> 10) & 0x1fu, mantissa = bits & 0x3ffu;
			uint32_t x;
			if (exponent == 0x1f)
				x = sign | 0x7f800000u | (mantissa << 13);
			else if (exponent != 0)
				x = sign | ((exponent + 112) << 23) | (mantissa << 13);
			else { // subnormal or zero
				const float value = static_cast<float>(mantissa) / 16777216.0f;
				std::memcpy(&x, &value, sizeof(x));
				x |= sign;
			}
			float value;
			std::memcpy(&value, &x, sizeof(value));
			return value;
		}
	};

	/**
	 * @brief The DLPack data type of the element type @p T: float32, bfloat16, float16, int32, or bool (stored as
	 * uint8_t, 0 or 1; e.g. masks).
	 */
	template <typename T>
	constexpr DLDataType dlDataType() {
		if constexpr (std::is_same_v<T, float>)
			return {kDLFloat, 32, 1};
		else if constexpr (std::is_same_v<T, bfloat16>)
			return {kDLBfloat, 16, 1};
		else if constexpr (std::is_same_v<T, float16>)
			return {kDLFloat, 16, 1};
		else if constexpr (std::is_same_v<T, int32_t>)
			return {kDLInt, 32, 1};
		else if constexpr (std::is_same_v<T, uint8_t>)
			return {kDLBool, 8, 1};
		else
			static_assert(sizeof(T) == 0, "unsupported element type");
	}

	/**
	 * @brief An owning, contiguous tensor on the CPU that can be passed to compiled functions (data aligned to
	 * kTensorAlignment). Use Tensor (float32) for `float` tensors and IntTensor (int32) for `int` tensors (e.g. token
	 * ids or labels).
	 */
	template <typename T>
	class BasicTensor {
	public:
		BasicTensor() : BasicTensor(std::vector<int64_t>{}) {}

		explicit BasicTensor(std::vector<int64_t> shape, T value = T{})
				: shape_(std::move(shape)), data_(numel(shape_), value) {
			updateView();
		}

		BasicTensor(std::vector<int64_t> shape, const std::vector<T>& data)
				: shape_(std::move(shape)), data_(data.begin(), data.end()) {
			if (data_.size() != numel(shape_))
				throw std::invalid_argument("mdl::runtime::Tensor: data size does not match the shape");
			updateView();
		}

		BasicTensor(const BasicTensor& other) : shape_(other.shape_), data_(other.data_) { updateView(); }
		BasicTensor(BasicTensor&& other) noexcept : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {
			updateView();
		}
		BasicTensor& operator=(BasicTensor other) noexcept {
			std::swap(shape_, other.shape_);
			std::swap(data_, other.data_);
			updateView();
			return *this;
		}

		/** @brief The DLPack view of this tensor (valid as long as the tensor is alive and not reassigned). */
		DLTensor* dl() { return &view_; }
		const DLTensor* dl() const { return &view_; }
		operator DLTensor*() { return &view_; }
		operator const DLTensor*() const { return &view_; }

		const std::vector<int64_t>& shape() const { return shape_; }
		std::span<T> values() { return data_; }
		std::span<const T> values() const { return data_; }
		T* data() { return data_.data(); }
		const T* data() const { return data_.data(); }
		size_t size() const { return data_.size(); }

	private:
		std::vector<int64_t> shape_;
		std::vector<T, AlignedAllocator<T>> data_;
		DLTensor view_{};

		static size_t numel(const std::vector<int64_t>& shape) {
			return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
		}

		void updateView() {
			view_ = DLTensor{
					.data = data_.data(),
					.device = {kDLCPU, 0},
					.ndim = static_cast<int32_t>(shape_.size()),
					.dtype = dlDataType<T>(),
					.shape = shape_.data(),
					.strides = nullptr,
					.byte_offset = 0,
			};
		}
	};

	using Tensor = BasicTensor<float>;
	using BFloat16Tensor = BasicTensor<bfloat16>;
	using Float16Tensor = BasicTensor<float16>;
	using IntTensor = BasicTensor<int32_t>;
	/** @brief A boolean tensor, e.g. the mask of a masked function (1: valid, 0: masked). */
	using MaskTensor = BasicTensor<uint8_t>;

	// ── safetensors ─────────────────────────────────────────────────────────────────────────────────────────────
	// File layout: an 8-byte little-endian header size N, an N-byte JSON header {name: {dtype, shape, data_offsets}}
	// (plus an optional "__metadata__" entry), and the tensor data (offsets relative to the end of the header).

	/** @brief One tensor of a safetensors file. */
	struct SafetensorsEntry {
		std::string dtype; ///< e.g. "F32", "I32"
		std::vector<int64_t> shape;
		uint64_t begin = 0, end = 0; ///< byte offsets of the data relative to the start of the data section
	};

	/** @brief Maps a tensor name of a file onto a weight path of the model; an empty result skips the tensor. */
	using NameMap = std::function<std::string(std::string_view)>;

	/** @brief Whether loading may convert float weights to the model's float type (e.g. a float32 file for bfloat16). */
	enum class FloatConversion { none, allowed };

	namespace detail {
		/** @brief The safetensors dtype of a DLPack data type (empty if unsupported). */
		inline std::string safetensorsDtype(DLDataType t) {
			if (t.lanes != 1)
				return {};
			if (t.code == kDLFloat && t.bits == 32)
				return "F32";
			if (t.code == kDLBfloat && t.bits == 16)
				return "BF16";
			if (t.code == kDLFloat && t.bits == 16)
				return "F16";
			if (t.code == kDLInt && t.bits == 32)
				return "I32";
			if (t.code == kDLUInt && t.bits == 8)
				return "U8";
			if (t.code == kDLBool && t.bits == 8)
				return "BOOL";
			return {};
		}

		/** @brief An opened safetensors file: its header and the stream positioned anywhere. */
		struct SafetensorsFile {
			std::ifstream in;
			std::map<std::string, SafetensorsEntry> entries;
			uint64_t dataStart = 0;
			uint64_t fileSize = 0;

			explicit SafetensorsFile(const std::filesystem::path& path) : in(path, std::ios::binary) {
				if (!in)
					throw std::runtime_error("safetensors: cannot open " + path.string());
				in.seekg(0, std::ios::end);
				fileSize = static_cast<uint64_t>(in.tellg());
				in.seekg(0);
				uint64_t headerSize = 0;
				in.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize)); // little endian
				if (!in || headerSize > fileSize - sizeof(headerSize))
					throw std::runtime_error("safetensors: " + path.string() + " is truncated");
				std::string header(headerSize, '\0');
				in.read(header.data(), static_cast<std::streamsize>(headerSize));
				dataStart = sizeof(headerSize) + headerSize;
				const auto json = nlohmann::json::parse(header);
				for (const auto& [name, meta] : json.items()) {
					if (name == "__metadata__")
						continue;
					SafetensorsEntry e{
							.dtype = meta.at("dtype").get<std::string>(),
							.shape = meta.at("shape").get<std::vector<int64_t>>(),
							.begin = meta.at("data_offsets").at(0).get<uint64_t>(),
							.end = meta.at("data_offsets").at(1).get<uint64_t>(),
					};
					if (e.begin > e.end || dataStart + e.end > fileSize)
						throw std::runtime_error("safetensors: tensor " + name + " is out of bounds");
					entries.emplace(name, std::move(e));
				}
			}

			/** @brief Reads the data of @p e into @p dst (exactly e.end - e.begin bytes). */
			void read(const SafetensorsEntry& e, void* dst) {
				in.seekg(static_cast<std::streamoff>(dataStart + e.begin));
				in.read(static_cast<char*>(dst), static_cast<std::streamsize>(e.end - e.begin));
				if (!in)
					throw std::runtime_error("safetensors: read failed");
			}
		};

		template <typename T>
		std::string safetensorsDtypeOf() {
			return safetensorsDtype(dlDataType<T>());
		}

		template <typename T>
		std::map<std::string, BasicTensor<T>> readSafetensors(const std::filesystem::path& path, bool skipOthers) {
			SafetensorsFile file{path};
			std::map<std::string, BasicTensor<T>> tensors;
			for (const auto& [name, entry] : file.entries) {
				if (entry.dtype != safetensorsDtypeOf<T>()) {
					if (skipOthers)
						continue;
					throw std::runtime_error("safetensors: tensor " + name + " has unsupported dtype " + entry.dtype);
				}
				BasicTensor<T> t{entry.shape};
				if (t.size() * sizeof(T) != entry.end - entry.begin)
					throw std::runtime_error("safetensors: tensor " + name + " has an inconsistent size");
				file.read(entry, t.data());
				tensors.emplace(name, std::move(t));
			}
			return tensors;
		}
	} // namespace detail

	/** @brief The entries of a safetensors file (names, dtypes, shapes) without reading the data. */
	inline std::map<std::string, SafetensorsEntry> readSafetensorsHeader(const std::filesystem::path& path) {
		return detail::SafetensorsFile{path}.entries;
	}

	/** @brief Loads all tensors of a safetensors file (all must be float32). */
	inline std::map<std::string, Tensor> loadSafetensors(const std::filesystem::path& path) {
		return detail::readSafetensors<float>(path, /*skipOthers=*/false);
	}

	/**
	 * @brief Loads the tensors of element type T (float or int32_t) of a safetensors file; tensors of other types are
	 * skipped (e.g. to read the float weights and the int32 token ids of one file separately).
	 */
	template <typename T>
	std::map<std::string, BasicTensor<T>> loadSafetensorsAs(const std::filesystem::path& path) {
		return detail::readSafetensors<T>(path, /*skipOthers=*/true);
	}

	/**
	 * @brief Writes tensors (contiguous, CPU) to a safetensors file, in the given order.
	 * @param tensors (name, tensor) pairs; the dtypes float32, int32, and uint8 are supported.
	 */
	inline void saveSafetensors(
			const std::filesystem::path& path, const std::vector<std::pair<std::string, const DLTensor*>>& tensors
	) {
		nlohmann::ordered_json header = nlohmann::ordered_json::object();
		uint64_t offset = 0;
		std::vector<uint64_t> sizes;
		for (const auto& [name, t] : tensors) {
			const std::string dtype = detail::safetensorsDtype(t->dtype);
			if (dtype.empty())
				throw std::runtime_error("safetensors: unsupported dtype of tensor " + name);
			if (t->strides != nullptr)
				throw std::runtime_error("safetensors: tensor " + name + " is not contiguous");
			std::vector<int64_t> shape(t->shape, t->shape + t->ndim);
			uint64_t bytes = (t->dtype.bits / 8) * std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<>{});
			header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + bytes}}};
			sizes.push_back(bytes);
			offset += bytes;
		}
		std::string json = header.dump();
		json.append((8 - (json.size() % 8)) % 8, ' '); // the data section starts 8-byte aligned
		std::ofstream out(path, std::ios::binary);
		if (!out)
			throw std::runtime_error("safetensors: cannot write " + path.string());
		const uint64_t headerSize = json.size();
		out.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
		out.write(json.data(), static_cast<std::streamsize>(json.size()));
		for (size_t i = 0; i < tensors.size(); ++i) {
			const DLTensor* t = tensors[i].second;
			out.write(static_cast<const char*>(t->data) + t->byte_offset, static_cast<std::streamsize>(sizes[i]));
		}
		if (!out)
			throw std::runtime_error("safetensors: write failed for " + path.string());
	}

	// ── weights of compiled models ──────────────────────────────────────────────────────────────────────────────
	// `Model` is a struct of a generated C++ header (e.g. mdl::models::Bert): `Model::Weights` has one DLTensor slot per
	// weight, `Model::Weights::infos` lists their paths, shapes, and dtypes, and `find(path)` returns a slot.

	/** @brief The weights of a model loaded from a file: the bound slots (`weights`) and the memory they point to. */
	template <typename Model>
	struct LoadedWeights {
		typename Model::Weights weights;
		/** @brief Tensors of the file that are not weights of the model (after the name mapping). */
		std::vector<std::string> unused;

		/** @brief The memory the slots point to (map nodes do not move, so moving LoadedWeights keeps them valid). */
		std::map<std::string, BasicTensor<float>> floats;
		std::map<std::string, BasicTensor<bfloat16>> bfloats;
		std::map<std::string, BasicTensor<float16>> halves;
		std::map<std::string, BasicTensor<int32_t>> ints;

		LoadedWeights() = default;
		LoadedWeights(const LoadedWeights&) = delete; // the slots point into this object's tensors
		LoadedWeights(LoadedWeights&&) = default;
		LoadedWeights& operator=(const LoadedWeights&) = delete;
		LoadedWeights& operator=(LoadedWeights&&) = default;
	};

	/**
	 * @brief Loads the weights of @p Model from a safetensors file and binds them.
	 * @details Every weight of the model must be in the file with the declared dtype and shape (dimensions that are only
	 * known at runtime match any size); all problems are reported together. Tensors of the file that are not weights of
	 * the model are listed in LoadedWeights::unused.
	 * @param map optional mapping of the file's tensor names onto the model's weight paths (e.g. from a Hugging Face
	 * checkpoint); an empty result skips the tensor.
	 * @param conversion whether float weights of another float type (F32, BF16, F16) are converted to the model's type
	 * (e.g. a float32 checkpoint for a model compiled with `--float bf16`); by default the types must agree.
	 */
	template <typename Model>
	LoadedWeights<Model> loadWeights(
			const std::filesystem::path& path, const NameMap& map = {}, FloatConversion conversion = FloatConversion::none
	) {
		detail::SafetensorsFile file{path};
		LoadedWeights<Model> result;
		std::map<std::string, const SafetensorsEntry*> byPath;
		for (const auto& [name, entry] : file.entries) {
			std::string target = map ? map(name) : name;
			if (target.empty())
				continue;
			if (result.weights.find(target) == nullptr) {
				result.unused.push_back(name);
				continue;
			}
			byPath.emplace(std::move(target), &entry);
		}

		std::string problems;
		for (const auto& info : Model::Weights::infos) {
			auto it = byPath.find(info.name);
			if (it == byPath.end()) {
				problems += std::string("\n  missing: ") + info.name;
				continue;
			}
			const SafetensorsEntry& e = *it->second;
			const std::string dtype = detail::safetensorsDtype(info.dtype);
			bool shapeOk = static_cast<int32_t>(e.shape.size()) == info.ndim;
			for (int32_t d = 0; shapeOk && d < info.ndim; ++d)
				shapeOk = info.shape[d] < 0 || info.shape[d] == e.shape[d];
			auto isFloat = [](const std::string& d) { return d == "F32" || d == "BF16" || d == "F16"; };
			const bool convert = conversion == FloatConversion::allowed && e.dtype != dtype && isFloat(e.dtype) && isFloat(dtype);
			if ((e.dtype != dtype && !convert) || !shapeOk) {
				std::string fileShape, modelShape;
				for (auto d : e.shape)
					fileShape += (fileShape.empty() ? "" : ", ") + std::to_string(d);
				for (int32_t d = 0; d < info.ndim; ++d)
					modelShape += (d > 0 ? ", " : "") + std::to_string(info.shape[d]);
				problems += std::string("\n  ") + info.name + ": file has " + e.dtype + "[" + fileShape + "], model expects " +
							dtype + "[" + modelShape + "]";
				continue;
			}
			auto bind = [&](auto& storage, auto tensor) {
				if (convert) {
					// Read in the file's type, convert each element (through float32).
					std::vector<char> raw(e.end - e.begin);
					file.read(e, raw.data());
					auto element = [&](size_t i) -> float {
						if (e.dtype == "F32") {
							float v;
							std::memcpy(&v, raw.data() + 4 * i, 4);
							return v;
						}
						uint16_t bits;
						std::memcpy(&bits, raw.data() + 2 * i, 2);
						return e.dtype == "BF16" ? static_cast<float>(std::bit_cast<bfloat16>(bits))
												 : static_cast<float>(std::bit_cast<float16>(bits));
					};
					for (size_t i = 0; i < tensor.size(); ++i)
						tensor.data()[i] = element(i);
				} else {
					file.read(e, tensor.data());
				}
				auto& stored = storage.emplace(info.name, std::move(tensor)).first->second;
				*result.weights.find(info.name) = stored.dl();
			};
			if (dtype == "I32")
				bind(result.ints, BasicTensor<int32_t>{e.shape});
			else if (dtype == "BF16")
				bind(result.bfloats, BasicTensor<bfloat16>{e.shape});
			else if (dtype == "F16")
				bind(result.halves, BasicTensor<float16>{e.shape});
			else
				bind(result.floats, BasicTensor<float>{e.shape});
		}
		if (!problems.empty())
			throw std::runtime_error("loading the weights from " + path.string() + " failed:" + problems);
		return result;
	}

	/** @brief Writes the (bound) weights of @p Model to a safetensors file, in canonical order. */
	template <typename Model>
	void saveWeights(const std::filesystem::path& path, const typename Model::Weights& weights) {
		std::vector<std::pair<std::string, const DLTensor*>> tensors;
		auto& slots = const_cast<typename Model::Weights&>(weights);
		for (const auto& info : Model::Weights::infos) {
			const DLTensor* t = *slots.find(info.name);
			if (t == nullptr)
				throw std::runtime_error(std::string("saveWeights: weight ") + info.name + " is not bound");
			tensors.emplace_back(info.name, t);
		}
		saveSafetensors(path, tensors);
	}

} // namespace mdl::runtime

#endif
