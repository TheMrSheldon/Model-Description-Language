#include <mdl/computegraphv2/datatype.hpp>

#include <mdl/computegraphv2/graph.hpp>
#include <mdl/computegraphv2/node.hpp>

#include <cassert>
#include <map>

using namespace mdl;

std::expected<DataType, std::string> mdl::setTypeArgs(const PrimitiveType& prim, const std::vector<TypeArg>& typeargs) {
	TensorType::Shape shape;
	for (auto& arg : typeargs) {
		using RT = std::expected<TensorType::Shape::value_type, std::string>;
		auto val = std::visit(
				overloaded{
						[](Elipsis) -> RT { return Elipsis{}; },
						[](const Atomic& value) -> RT {
							if (auto val = std::get_if<int>(&value)) {
								return static_cast<unsigned>(*val);
							} else {
								return std::unexpected("Invalid type"); /** \todo more descriptive message **/
							}
						},
						[](const Node* node) -> RT { return node; }
				},
				arg
		);
		if (val.has_value())
			shape.emplace_back(*val);
		else
			return std::unexpected(val.error());
	}
	return TensorType{.type = prim, .shape = std::move(shape)};
}

std::expected<DataType, std::string> mdl::setTypeArgs(const TensorType& tensor, const std::vector<TypeArg>& typeargs) {
	return std::unexpected("Type arguments are not available for tensor types");
}

std::expected<DataType, std::string> mdl::setTypeArgs(const AggregateType& agg, const std::vector<TypeArg>& typeargs) {
	return std::unexpected("Type arguments are not available for aggregate types");
}

std::expected<DataType, std::string>
mdl::setTypeArgs(const BoundModuleType& mod, const std::vector<TypeArg>& typeargs) {
	const auto& decl = mod.definition.get();
	if (typeargs.size() != decl.typeargs.size())
		return std::unexpected(std::format("Expected {} type arg(s), got {}", decl.typeargs.size(), typeargs.size()));
	BoundModuleType bound{mod.definition, {}};
	for (size_t i = 0; i < typeargs.size(); ++i) {
		auto& [name, _] = decl.typeargs[i];
		if (auto atom = std::get_if<Atomic>(&typeargs[i]))
			bound.hyperparamValues.emplace_back(name, *atom);
		else
			return std::unexpected("Module hyperparameters must be compile-time constants");
	}
	return bound;
}

std::expected<DataType, std::string> mdl::setTypeArgs(const DataType& dtype, const std::vector<TypeArg>& typeargs) {
	return std::visit([&](auto& d) { return setTypeArgs(d, typeargs); }, dtype);
}

FunctionDecl& mdl::getMember(const DataType& dtype, std::string_view name, const std::vector<DataType> argtypes) {
	/** \todo implement properly for all types; for now only TensorType arithmetic operators are supported **/
	assert(std::holds_alternative<TensorType>(dtype));

	using mdl::builtins::FunctionIDs;
	static Computegraph graph;
	static std::map<std::string, FunctionDecl, std::less<>> ops = []() {
		std::map<std::string, FunctionDecl, std::less<>> m;
		DataType ttype = TensorType{.type = PrimitiveType::floating, .shape = {Elipsis{}}};

		auto makeOp = [&](std::string opName, FunctionIDs id) {
			const Node& lhs = graph.addParameterNode(ttype);
			const Node& rhs = graph.addParameterNode(ttype);
			const Node& out = graph.addInvocationNode(id, ttype, {std::cref(lhs), std::cref(rhs)});
			m.emplace(
					std::move(opName), FunctionDecl{
											   .name = opName,
											   .templateParameters = {},
											   .parameters =
													   {{"lhs", std::cref(std::get<ParameterNode>(lhs))},
														{"rhs", std::cref(std::get<ParameterNode>(rhs))}},
											   .output = std::cref(out),
									   }
			);
		};

		makeOp("operator+", FunctionIDs::TENSOR_ADD);
		makeOp("operator-", FunctionIDs::TENSOR_SUB);
		makeOp("operator*", FunctionIDs::TENSOR_DOT);
		makeOp("operator@", FunctionIDs::TENSOR_PROD);
		makeOp("operator/", FunctionIDs::TENSOR_DIV);
		return m;
	}();

	auto it = ops.find(name);
	assert(it != ops.end()); /** \todo proper error reporting for unsupported operators **/
	return it->second;
}