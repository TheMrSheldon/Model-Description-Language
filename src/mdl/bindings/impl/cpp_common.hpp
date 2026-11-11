#pragma once

#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/exports.hpp>
#include <mdl/utils/overload.hpp>

#include <string>

namespace mdl::bindings::detail {

	inline std::string primTypeStr(mdl::PrimitiveType p) {
		switch (p) {
		case mdl::PrimitiveType::floating:
			return "float"; /** \todo load the configured datatype for default float */
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
						[](const mdl::BoundModuleType&) -> std::string { return std::string{"module"}; }
				},
				dt
		);
	}

	/// Shape of an exported tensor, e.g. `float[B, 768]` (`B` marks dimensions only known at runtime).
	inline std::string shapeStr(const mdl::TensorSignature& sig) {
		std::string s = primTypeStr(sig.dtype) + "[";
		for (size_t i = 0; i < sig.shape.size(); ++i)
			s += (i > 0 ? ", " : "") + (sig.shape[i] < 0 ? std::string{"B"} : std::to_string(sig.shape[i]));
		return s + "]";
	}

} // namespace mdl::bindings::detail
