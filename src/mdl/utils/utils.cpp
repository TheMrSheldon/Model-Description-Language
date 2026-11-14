#include <mdl/utils/utils.hpp>

#include <mdl/computegraphv2/node.hpp>

#include <algorithm>
#include <ranges>
#include <utility>

using mdl::AggregateType;
using mdl::DataType;
using mdl::Elipsis;
using mdl::PrimitiveType;
using mdl::TensorType;

using namespace std::literals;

#if !__cpp_lib_to_underlying
namespace std {
	template <class Enum>
	constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept {
		return static_cast<std::underlying_type_t<Enum>>(e);
	}
} // namespace std
#endif

namespace std {
	std::string to_string(const Elipsis&) { return "..."; }
	std::string to_string(const mdl::Node* node) {
		return node ? "<Node>" : "<Null>"; /** \todo possibly improve conversion by writing the expression **/
	}
} // namespace std

std::string std::to_string(const PrimitiveType& prim) {
	switch (prim) {
	case PrimitiveType::error:
		return "<error>";
	case PrimitiveType::float32:
		return "float32";
	case PrimitiveType::integer:
		return "int";
	case PrimitiveType::uint:
		return "uint";
	case PrimitiveType::boolean:
		return "bool";
	case PrimitiveType::float16:
		return "float16";
	case PrimitiveType::bfloat16:
		return "bfloat16";
	case PrimitiveType::int8:
		return "int8";
	}
	return "<error>";
}
std::string std::to_string(const AggregateType& agg) { return "Aggregate Type"; /** \todo implement **/ }
std::string std::to_string(const TensorType& tensortype) {
	std::string args;
	auto r = tensortype.shape | std::views::transform([](const auto& x) {
				 return std::visit([](const auto& x) { return std::to_string(x); }, x);
			 }) |
			 std::views::join_with(", "sv);
	std::ranges::copy(std::begin(r), std::end(r), std::back_inserter(args));
	return std::to_string(tensortype.type) + "[" + args + "]";
}
std::string std::to_string(const DataType& dtype) {
	return std::visit([](auto x) { return std::to_string(x); }, dtype);
}