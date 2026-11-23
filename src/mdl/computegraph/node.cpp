#include <mdl/computegraphv2/node.hpp>

#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/graph.hpp>
#include <mdl/utils/overload.hpp>

const mdl::DataType& mdl::ConstNode::datatype() const {
	static DataType inttype = PrimitiveType::integer;
	static DataType floattype = PrimitiveType::float32;
	return std::visit(
			mdl::overloaded{
					[&](const int&) -> const DataType& { return inttype; },
					[&](const double&) -> const DataType& { return floattype; },
					[&](const std::vector<int>& v) -> const DataType& {
						if (!_cachedType)
							_cachedType = TensorType{PrimitiveType::integer, {static_cast<unsigned>(v.size())}};
						return *_cachedType;
					},
			},
			value
	);
}

#include <vector>

using namespace mdl;

const DataType& mdl::getDataType(const ConstNode& node) { return node.datatype(); }
const DataType& mdl::getDataType(const InvocationNode& node) { return node.datatype(); }
const DataType& mdl::getDataType(const ErrorNode& node) { return node.datatype(); }
const DataType& mdl::getDataType(const InputNode& node) { return node.datatype(); }
const DataType& mdl::getDataType(const ParameterNode& node) { return node.datatype(); }
const DataType& mdl::getDataType(const HyperparameterNode& node) { return node.datatype(); }
const DataType& mdl::getDataType(const Node& node) {
	return std::visit([](const auto& node) -> const DataType& { return getDataType(node); }, node);
}

const std::string& mdl::getName(const mdl::InputNode& node) { return node.name; }
const std::string& mdl::getName(const mdl::ParameterNode& node) { return node.name; }
const std::string& mdl::getName(const mdl::HyperparameterNode& node) { return node.name; }
std::expected<std::reference_wrapper<const std::string>, bool> mdl::getName(const Node& node) {
	using RT = std::expected<std::reference_wrapper<const std::string>, bool>;
	return std::visit(
			mdl::overloaded{
					[](const mdl::InputNode& node) -> RT { return std::cref(node.name); },
					[](const mdl::ParameterNode& node) -> RT { return std::cref(node.name); },
					[](const mdl::HyperparameterNode& node) -> RT { return std::cref(node.name); },
					[](const auto&) -> RT { return std::unexpected(false); },
			},
			node
	);
}

bool mdl::isCompiletimeEvalable(const Node& node) {
	return std::visit([](const auto& node) { return node.isConsteval(); }, node);
}