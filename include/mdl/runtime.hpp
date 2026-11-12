#ifndef MDL_RUNTIME_HPP
#define MDL_RUNTIME_HPP

/**
 * @file
 * @brief Small header-only helpers for programs that call compiled models: an owning (CPU, float32) tensor that can be
 * passed as `DLTensor*` and a loader for `.safetensors` files.
 */

#include <dlpack/dlpack.h>

#include <cctype>
#include <cstdint>
#include <cstring>
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

	/** @brief The DLPack data type of the element type @p T (float32 or int32). */
	template <typename T>
	constexpr DLDataType dlDataType() {
		if constexpr (std::is_same_v<T, float>)
			return {kDLFloat, 32, 1};
		else if constexpr (std::is_same_v<T, int32_t>)
			return {kDLInt, 32, 1};
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
	using IntTensor = BasicTensor<int32_t>;

	namespace detail {
		/** @brief Minimal reader for the (flat) JSON header of a safetensors file. */
		class SafetensorsHeaderParser {
		public:
			explicit SafetensorsHeaderParser(std::string_view json) : json(json) {}

			struct Entry {
				std::string dtype;
				std::vector<int64_t> shape;
				std::pair<uint64_t, uint64_t> offsets{0, 0};
			};

			std::map<std::string, Entry> parse() {
				std::map<std::string, Entry> entries;
				expect('{');
				while (!consume('}')) {
					std::string name = string();
					expect(':');
					if (name == "__metadata__")
						skipValue();
					else
						entries.emplace(std::move(name), entry());
					consume(',');
				}
				return entries;
			}

		private:
			std::string_view json;
			size_t pos = 0;

			void skipWs() {
				while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])))
					++pos;
			}
			bool consume(char c) {
				skipWs();
				if (pos < json.size() && json[pos] == c) {
					++pos;
					return true;
				}
				return false;
			}
			void expect(char c) {
				if (!consume(c))
					throw std::runtime_error(std::string{"safetensors: malformed header, expected '"} + c + "'");
			}
			std::string string() {
				expect('"');
				std::string s;
				while (pos < json.size() && json[pos] != '"') {
					if (json[pos] == '\\')
						++pos;
					s += json[pos++];
				}
				expect('"');
				return s;
			}
			int64_t number() {
				skipWs();
				size_t end = pos;
				while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) || json[end] == '-'))
					++end;
				int64_t value = std::stoll(std::string{json.substr(pos, end - pos)});
				pos = end;
				return value;
			}
			std::vector<int64_t> numbers() {
				std::vector<int64_t> values;
				expect('[');
				while (!consume(']')) {
					values.push_back(number());
					consume(',');
				}
				return values;
			}
			void skipValue() {
				skipWs();
				int depth = 0;
				bool inString = false;
				for (; pos < json.size(); ++pos) {
					char c = json[pos];
					if (inString) {
						if (c == '\\')
							++pos;
						else if (c == '"')
							inString = false;
					} else if (c == '"') {
						inString = true;
					} else if (c == '{' || c == '[') {
						++depth;
					} else if (c == '}' || c == ']') {
						if (--depth == 0) {
							++pos;
							return;
						}
					} else if (depth == 0 && c == ',') {
						return;
					}
				}
			}
			Entry entry() {
				Entry e;
				expect('{');
				while (!consume('}')) {
					std::string key = string();
					expect(':');
					if (key == "dtype") {
						e.dtype = string();
					} else if (key == "shape") {
						e.shape = numbers();
					} else if (key == "data_offsets") {
						auto o = numbers();
						if (o.size() != 2)
							throw std::runtime_error("safetensors: malformed data_offsets");
						e.offsets = {static_cast<uint64_t>(o[0]), static_cast<uint64_t>(o[1])};
					} else {
						skipValue();
					}
					consume(',');
				}
				return e;
			}
		};
	} // namespace detail

	/**
	 * @brief Loads all float32 tensors of a `.safetensors` file.
	 * @throws std::runtime_error if the file cannot be read or contains tensors of another data type.
	 */
	inline std::map<std::string, Tensor> loadSafetensors(const std::string& path) {
		std::ifstream in{path, std::ios::binary};
		if (!in)
			throw std::runtime_error("cannot open " + path);
		uint64_t headerSize = 0;
		in.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize)); // little endian
		std::string header(headerSize, '\0');
		in.read(header.data(), static_cast<std::streamsize>(headerSize));
		std::vector<char> payload{std::istreambuf_iterator<char>(in), {}};

		std::map<std::string, Tensor> tensors;
		for (auto& [name, entry] : detail::SafetensorsHeaderParser{header}.parse()) {
			if (entry.dtype != "F32")
				throw std::runtime_error("safetensors: tensor " + name + " has unsupported dtype " + entry.dtype);
			auto [begin, end] = entry.offsets;
			if (end > payload.size() || begin > end)
				throw std::runtime_error("safetensors: tensor " + name + " is out of bounds");
			std::vector<float> data((end - begin) / sizeof(float));
			std::memcpy(data.data(), payload.data() + begin, end - begin);
			tensors.emplace(name, Tensor{std::move(entry.shape), std::move(data)});
		}
		return tensors;
	}

} // namespace mdl::runtime

#endif
