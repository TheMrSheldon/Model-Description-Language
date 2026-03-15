#pragma once

#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>
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
						[](const mdl::BoundModuleType&) -> std::string { return std::string{"module"}; }
				},
				dt
		);
	}

	/// C++ parameter type for the TVM FFI: tensors become NDArray, scalars their C++ equivalents.
	inline std::string cppParamType(const mdl::DataType& dt) {
		return std::visit(
				mdl::overloaded{
						[](const mdl::PrimitiveType& p) -> std::string {
							switch (p) {
							case mdl::PrimitiveType::floating:
								return std::string{"float"};
							case mdl::PrimitiveType::integer:
								return std::string{"std::int32_t"};
							case mdl::PrimitiveType::uint:
								return std::string{"std::uint32_t"};
							case mdl::PrimitiveType::boolean:
								return std::string{"bool"};
							default:
								return std::string{"void*"};
							}
						},
						[](const mdl::TensorType&) { return std::string{"tvm::runtime::NDArray"}; },
						[](const mdl::AggregateType&) { return std::string{"void*"}; },
						[](const mdl::BoundModuleType&) { return std::string{"void*"}; },
				},
				dt
		);
	}

	inline std::string funcSignatureStr(const mdl::FunctionDecl& func) {
		std::string sig = func.name + "(";
		bool first = true;
		for (const auto& [pname, pnode] : func.parameters) {
			if (!first)
				sig += ", ";
			sig += pname + ": " + dtypeStr(mdl::getDataType(pnode.get()));
			first = false;
		}
		return sig + ") -> " + dtypeStr(func.returnType());
	}

} // namespace mdl::bindings::detail
