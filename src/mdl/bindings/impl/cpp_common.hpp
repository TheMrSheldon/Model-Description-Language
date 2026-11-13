#pragma once

#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/exports.hpp>
#include <mdl/utils/overload.hpp>

#include <algorithm>
#include <format>
#include <string>

namespace mdl::bindings::detail {

	inline std::string primTypeStr(mdl::PrimitiveType p) {
		switch (p) {
		case mdl::PrimitiveType::float32:
			return "float32";
		case mdl::PrimitiveType::float16:
			return "float16";
		case mdl::PrimitiveType::bfloat16:
			return "bfloat16";
		case mdl::PrimitiveType::integer:
			return "int"; /** \todo load the configured datatype for default int */
		case mdl::PrimitiveType::uint:
			return "uint"; /** \todo load the configured datatype for default uint */
		case mdl::PrimitiveType::boolean:
			return "bool";
		default:
			return "error";
		}
	}

	/// Human-readable type string used in doc-comments for all targets.
	inline std::string dtypeStr(const mdl::DataType& dt) {
		return std::visit(
				mdl::overloaded{
						[](const mdl::PrimitiveType& p) -> std::string { return primTypeStr(p); },
						[](const mdl::TensorType& t) -> std::string {
							std::string s = primTypeStr(t.type);
							for (const auto& dim : t.shape) {
								s += "[";
								std::visit(
										mdl::overloaded{
												[&s](unsigned u) { s += std::to_string(u); },
												[&s](mdl::Elipsis) { s += "..."; }, [&s](const mdl::Node*) { s += "?"; }
										},
										dim
								);
								s += "]";
							}
							return s;
						},
						[](const mdl::AggregateType&) -> std::string { return std::string{"struct"}; },
						[](const mdl::ModuleType&) -> std::string { return std::string{"module"}; },
						[](const mdl::BoundModuleType&) -> std::string { return std::string{"module"}; },
						[](const mdl::ArrayType& a) -> std::string {
							return std::format("array[{}; {}]", dtypeStr(*a.element), a.size);
						}
				},
				dt
		);
	}

	/// Shape of an exported tensor, e.g. `float[B, 768]` or `float[B0, B1, 768]` (`B`/`Bi` mark the runtime
	/// dimensions of the ellipsis; they are the same in all tensors of a function).
	inline std::string shapeStr(const mdl::TensorSignature& sig) {
		const auto dynamic = std::count(sig.shape.begin(), sig.shape.end(), int64_t{-1});
		std::string s = primTypeStr(sig.dtype) + "[";
		int k = 0;
		for (size_t i = 0; i < sig.shape.size(); ++i) {
			s += i > 0 ? ", " : "";
			if (sig.shape[i] >= 0)
				s += std::to_string(sig.shape[i]);
			else
				s += dynamic == 1 ? std::string{"B"} : "B" + std::to_string(k++);
		}
		return s + "]";
	}

} // namespace mdl::bindings::detail
