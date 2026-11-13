#include <mdl/exports.hpp>

#include "./argvals.hpp"
#include "./logging.hpp"

#include <mdl/compiler.hpp>
#include <mdl/computegraphv2/function.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/overload.hpp>
#include <mdl/passes/masking.hpp>
#include <mdl/passes/shapes.hpp>

#include <algorithm>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace mdl;

std::optional<int64_t> mdl::evalConstant(const Node& node, const ArgVals& argvals) {
	using FIDs = builtins::FunctionIDs;
	return std::visit(
			overloaded{
					[](const ConstNode& cn) -> std::optional<int64_t> {
						if (const int* v = std::get_if<int>(&cn.value))
							return *v;
						return std::nullopt;
					},
					[&](const HyperparameterNode& hn) -> std::optional<int64_t> {
						if (auto v = argvals.lookup(hn.name))
							return *v;
						return std::nullopt;
					},
					[&](const InvocationNode& inv) -> std::optional<int64_t> {
						if (inv.arguments.size() != 2)
							return std::nullopt;
						auto lhs = evalConstant(inv.arguments[0].get(), argvals);
						auto rhs = evalConstant(inv.arguments[1].get(), argvals);
						if (!lhs || !rhs)
							return std::nullopt;
						switch (inv.id) {
						case FIDs::TENSOR_ADD:
							return *lhs + *rhs;
						case FIDs::TENSOR_SUB:
							return *lhs - *rhs;
						case FIDs::TENSOR_PROD:
							return *lhs * *rhs;
						case FIDs::TENSOR_DIV:
							return *rhs != 0 ? std::optional<int64_t>{*lhs / *rhs} : std::nullopt;
						default:
							return std::nullopt;
						}
					},
					[](const auto&) -> std::optional<int64_t> { return std::nullopt; },
			},
			node
	);
}

static TensorSignature makeSignature(
		std::string name, const DataType& dtype, const Node* node, const ArgVals& argvals, unsigned ellipsisRank
) {
	TensorSignature sig{.name = std::move(name), .dtype = PrimitiveType::error, .shape = {}, .node = node};
	std::visit(
			overloaded{
					[&](const PrimitiveType& p) { sig.dtype = p; },
					[&](const TensorType& t) {
						sig.dtype = t.type;
						for (const auto& dim : t.shape) {
							if (std::holds_alternative<Elipsis>(dim)) {
								sig.shape.insert(sig.shape.end(), ellipsisRank, -1);
								continue;
							}
							sig.shape.push_back(std::visit(
									overloaded{
											[](unsigned u) -> int64_t { return u; },
											[](Elipsis) -> int64_t { return -1; },
											[&](const Node* n) -> int64_t { return evalConstant(*n, argvals).value_or(-1); },
									},
									dim
							));
						}
					},
					[](const auto&) {},
			},
			dtype
	);
	return sig;
}

/**
 * @brief The value of a hyperparameter of type @p type from the argument values at @p path (e.g. `Module.conf`):
 * a ConstNode for scalars and an AggregateInstance for structs; std::nullopt if a value is missing.
 */
static std::optional<Object>
hyperparameterValue(const CompilationUnit& unit, const DataType& type, const std::string& path, const ArgVals& argvals) {
	if (std::holds_alternative<PrimitiveType>(type)) {
		auto value = argvals.lookupScalar(path);
		if (!value)
			return std::nullopt;
		Atomic atomic = std::visit(
				overloaded{
						[](int64_t i) { return Atomic{static_cast<int>(i)}; },
						[](double d) { return Atomic{d}; },
						[](bool b) { return Atomic{b ? 1 : 0}; },
				},
				*value
		);
		return Object{&unit.graph().addConstNode(std::move(atomic))};
	}
	if (const auto* agg = std::get_if<AggregateType>(&type)) {
		std::map<std::string, Object> members;
		for (const auto& [name, memberType] : agg->members) {
			auto member = hyperparameterValue(unit, memberType, path + "." + name, argvals);
			if (!member)
				return std::nullopt;
			members.emplace(name, std::move(*member));
		}
		auto instance = AggregateInstance::create(*agg, std::move(members));
		return instance ? std::optional<Object>{Object{std::move(*instance)}} : std::nullopt;
	}
	return std::nullopt;
}

/** @brief Flattens a (concrete) hyperparameter value into (path, value) pairs. */
static void flattenHyperparameter(
		const std::string& path, const Object& value,
		std::vector<std::pair<std::string, std::variant<int64_t, double>>>& out
) {
	if (const auto* node = std::get_if<const Node*>(&value)) {
		if (const auto* cn = std::get_if<ConstNode>(*node)) {
			if (const auto* i = std::get_if<int>(&cn->value))
				out.emplace_back(path, int64_t{*i});
			else if (const auto* d = std::get_if<double>(&cn->value))
				out.emplace_back(path, *d);
		}
	} else if (const auto* agg = std::get_if<AggregateInstance>(&value)) {
		size_t i = 0;
		for (const auto& [name, _] : agg->getType().members)
			flattenHyperparameter(path + "." + name, agg->getMembers()[i++], out);
	}
}

std::vector<ExportedFunction> mdl::exportedFunctions(
		const CompilationUnit& unit, const ArgVals& argvals, unsigned ellipsisRank, const GradientRequest& gradients,
		const std::vector<std::string>& masked, const std::map<std::string, unsigned>& inputRanks
) {
	std::vector<ExportedFunction> exports;
	std::set<std::string> differentiated; // the requested gradient functions that some compiled module defines
	for (const auto& [moduleName, member] : unit.definitions) {
		const auto* definition = std::get_if<ModuleType>(&member);
		if (definition == nullptr)
			continue;
		// Modules with hyperparameters are instantiated with the values given in the argument values; modules whose
		// hyperparameters are not (all) given are not compiled.
		const ModuleType* module = definition;
		std::vector<std::pair<std::string, std::variant<int64_t, double>>> hyperparameters;
		if (!definition->instantiated) {
			std::map<std::string, Object> values;
			bool complete = true;
			for (const auto& [pname, hn] : definition->typeargs) {
				auto value = hyperparameterValue(unit, hn.get().type, moduleName + "." + pname, argvals);
				if (!value) {
					complete = false;
					break;
				}
				flattenHyperparameter(pname, *value, hyperparameters);
				values.emplace(pname, std::move(*value));
			}
			if (!complete)
				continue;
			auto result = unit.instantiate(moduleName, values);
			if (!result)
				throw std::runtime_error(result.error());
			module = &result->get();
		}
		const auto order = weightPaths(*module);

		// The member functions; the gradient pass replaces the requested ones by their gradient functions.
		struct Function {
			std::string name;
			const FunctionDecl* decl;
			std::optional<ContextBuffer> context = std::nullopt;
			size_t numPacked = 0; ///< packed outputs (producer: the last ones) or inputs (consumer: the first ones)
			size_t firstGradient = std::numeric_limits<size_t>::max(); ///< index of the first gradient output
			std::vector<std::string> gradientOf = {};				   ///< what the gradient outputs are gradients of
			std::map<std::string, shapes::Shape> cotangents = {};	   ///< the exact shapes of its cotangent inputs
		};
		std::vector<Function> functions;
		auto requested = [&](const std::string& name) {
			return std::find(gradients.functions.begin(), gradients.functions.end(), name) != gradients.functions.end();
		};
		auto evaluate = [&](const Node& n) { return evalConstant(n, argvals); };

		// The shapes of the inputs whose `...` has fewer dimensions (the leading ones of the function's `...`).
		// Exact shapes of the cotangent inputs of a gradient function by name (`d_<output>`: the output's shape).
		using CotangentShapes = std::map<std::string, shapes::Shape>;
		CotangentShapes cotangentShapes; // of the gradient function being built (see knownInputs)
		auto knownInputs = [&](const FunctionDecl& f) {
			std::unordered_map<const Node*, shapes::Shape> known;
			if (inputRanks.empty() && cotangentShapes.empty())
				return known;
			std::unordered_set<const Node*> seen;
			std::vector<const Node*> stack;
			for (const auto& out : f.outputs)
				stack.push_back(&out.get());
			while (!stack.empty()) {
				const Node* n = stack.back();
				stack.pop_back();
				if (!seen.insert(n).second)
					continue;
				if (const auto* inv = std::get_if<InvocationNode>(n)) {
					for (const auto& a : inv->arguments)
						stack.push_back(&a.get());
					continue;
				}
				const auto* in = std::get_if<InputNode>(n);
				if (in != nullptr)
					if (auto c = cotangentShapes.find(in->name); c != cotangentShapes.end()) {
						known.emplace(n, c->second);
						continue;
					}
				auto rank = in ? inputRanks.find(in->name) : inputRanks.end();
				const auto* t = in ? std::get_if<TensorType>(&in->type) : nullptr;
				if (rank == inputRanks.end() || t == nullptr)
					continue;
				shapes::Shape shape;
				for (const auto& dim : t->shape) {
					if (std::holds_alternative<Elipsis>(dim)) {
						for (unsigned k = 0; k < rank->second; ++k)
							shape.push_back(shapes::Dim::symbol(ellipsisRank == 1 ? std::string{"B"} : "B" + std::to_string(k)));
					} else if (const auto* u = std::get_if<unsigned>(&dim)) {
						shape.push_back(shapes::Dim::constant(*u));
					} else {
						auto v = evaluate(*std::get<const Node*>(dim));
						shape.push_back(v ? shapes::Dim::constant(*v) : shapes::Dim::unknown());
					}
				}
				known.emplace(n, std::move(shape));
			}
			return known;
		};

		// Masked functions replace the originals (also as the functions that are differentiated).
		std::map<std::string, const FunctionDecl*> decls;
		auto hasEllipsis = [](const DataType& type) {
			const auto* t = std::get_if<TensorType>(&type);
			return t != nullptr && std::any_of(t->shape.begin(), t->shape.end(), [](const auto& d) {
				return std::holds_alternative<Elipsis>(d);
			});
		};
		auto isMasked = [](const DataType& type) {
			const auto* t = std::get_if<TensorType>(&type);
			return t != nullptr && t->masked;
		};
		std::vector<std::string> maskedHere = masked; // requested, plus functions that declare masked parameters
		for (const auto& [funcName, decl] : module->functions) {
			decls.emplace(funcName, &decl);
			if (std::any_of(decl.parameters.begin(), decl.parameters.end(), [&](const auto& p) {
					return isMasked(p.second.get().type);
				}) && std::find(maskedHere.begin(), maskedHere.end(), funcName) == maskedHere.end())
				maskedHere.push_back(funcName);
		}
		for (const auto& name : maskedHere) {
			auto it = decls.find(name);
			if (it == decls.end())
				continue;
			// Elaborate the function for masked inputs (its `...` parameters), so that it calls masked overloads.
			std::vector<bool> pattern;
			for (const auto& [_, p] : it->second->parameters)
				pattern.push_back(isMasked(p.get().type) || hasEllipsis(p.get().type));
			it->second = &unit.specializeMasked(*it->second, pattern);
			auto shapes = shapes::infer(*it->second, ellipsisRank, evaluate, knownInputs(*it->second));
			if (!shapes)
				throw std::runtime_error(moduleName + "." + shapes.error());
			auto result = masking::apply(unit.graph(), *it->second, *shapes);
			if (!result)
				throw std::runtime_error(moduleName + "." + result.error());
			it->second = &unit.keep(std::move(result->function));
		}

		for (const auto& [funcName, decl] : decls)
			if (!requested(funcName))
				functions.push_back({funcName, decl});
		for (const auto& name : gradients.functions) {
			auto it = decls.find(name);
			if (it == decls.end())
				continue; // another module may define it (checked below)
			differentiated.insert(name);
			const FunctionDecl& original = *it->second;
			// Exact shapes for the compiled `...` rank, so that the gradient pass does not depend on imprecise types.
			shapes::Oracle oracle{ellipsisRank, evaluate, knownInputs(original)};
			// The cotangents `d_<output>` get the outputs' exact shapes (their `...` may have fewer dimensions).
			cotangentShapes.clear();
			if (autodiff::numCotangents(original) > 0)
				for (size_t i = 0; i < original.outputs.size(); ++i)
					if (const shapes::Shape* s = oracle.shapeOf(original.outputs[i].get()))
						cotangentShapes["d_" + (original.outputs.size() == 1 ? std::string{"output"} : "output" + std::to_string(i))] = *s;
			autodiff::Options options = gradients.options;
			options.shapes = &oracle;
			auto grad = autodiff::vjp(unit.graph(), original, order, options);
			if (!grad)
				throw std::runtime_error(moduleName + "." + grad.error());
			const FunctionDecl& vjp = unit.keep(std::move(*grad));

			// What the gradient outputs are gradients of (as selected by autodiff::vjp).
			std::unordered_set<const Node*> reachable;
			std::function<void(const Node&)> reach = [&](const Node& node) {
				if (!reachable.insert(&node).second)
					return;
				if (const auto* inv = std::get_if<InvocationNode>(&node))
					for (const auto& arg : inv->arguments)
						reach(arg.get());
			};
			for (const auto& output : original.outputs)
				reach(output.get());
			std::vector<std::string> gradientOf;
			for (const auto& path : order) {
				const bool selected =
						gradients.options.trainable.empty() || gradients.options.trainable.selects(path);
				const bool used = std::any_of(reachable.begin(), reachable.end(), [&](const Node* n) {
					const auto* pn = std::get_if<ParameterNode>(n);
					return pn != nullptr && pn->name == path;
				});
				if (selected && used)
					gradientOf.push_back(path);
			}
			auto floating = [](const DataType& type) { // integer inputs (e.g. token ids) have no gradient
				if (const auto* p = std::get_if<PrimitiveType>(&type))
					return isFloat(*p);
				const auto* t = std::get_if<TensorType>(&type);
				return t != nullptr && isFloat(t->type);
			};
			if (gradients.options.inputs)
				for (const auto& [pname, input] : original.parameters)
					if (std::any_of(reachable.begin(), reachable.end(), [&](const Node* n) {
							return std::get_if<InputNode>(n) == &input.get();
						}) && floating(input.get().type))
						gradientOf.push_back(pname);

			if (autodiff::numCotangents(original) == 0 || gradients.fused) {
				functions.push_back({name, &vjp, std::nullopt, 0, original.outputs.size(), gradientOf, cotangentShapes});
				continue;
			}
			const PathPatterns checkpoint{gradients.checkpoint};
			auto parts = autodiff::split(
					unit.graph(), vjp, original.outputs.size(), gradients.checkpoint.empty() ? nullptr : &checkpoint
			);
			if (!parts)
				throw std::runtime_error(moduleName + "." + parts.error());
			ContextBuffer context{
					.name = name + "_ctx",
					.sizeSymbol = mangledName(moduleName, name + "_ctx_size"),
					.producer = true,
					.ellipsisRank = ellipsisRank,
			};
			parts->train.name = name;
			functions.push_back({name, &unit.keep(std::move(parts->train)), context, parts->numResiduals});
			context.producer = false;
			functions.push_back(
					{name + "_backward", &unit.keep(std::move(parts->backward)), context, parts->numResiduals, 0,
					 gradientOf, cotangentShapes}
			);
		}

		for (const auto& function : functions) {
			const std::string& funcName = function.name;
			const FunctionDecl* declPtr = function.decl;
			cotangentShapes = function.cotangents; // (for knownInputs and the input signatures)
			const FunctionDecl& decl = *declPtr;
			if (!decl.templateParameters.empty())
				continue;
			ExportedFunction fn{
					.module = moduleName,
					.function = funcName,
					.symbol = mangledName(moduleName, funcName),
					.decl = &decl,
					.inputs = {},
					.weights = {},
					.outputs = {},
					.moduleWeights = order,
					.hyperparameters = hyperparameters,
					.context = function.context,
			};

			// Collect the graph nodes of the arguments and all weights reachable from the outputs.
			std::unordered_map<const void*, const Node*> inputNodes;
			std::vector<const Node*> weights;
			std::unordered_set<const Node*> visited;
			std::function<void(const Node&)> collect = [&](const Node& node) {
				if (!visited.insert(&node).second)
					return;
				if (const auto* inv = std::get_if<InvocationNode>(&node)) {
					for (const auto& arg : inv->arguments)
						collect(arg.get());
				} else if (const auto* in = std::get_if<InputNode>(&node)) {
					inputNodes[in] = &node;
				} else if (std::holds_alternative<ParameterNode>(node)) {
					weights.push_back(&node);
				}
			};
			for (const auto& output : decl.outputs)
				collect(output.get());

			for (const auto& [pname, input] : decl.parameters) {
				auto it = inputNodes.find(&input.get());
				auto rank = inputRanks.find(pname);
				bool smaller = rank != inputRanks.end() && rank->second < ellipsisRank;
				unsigned inputRank = smaller ? rank->second : ellipsisRank;
				if (auto c = cotangentShapes.find(pname); c != cotangentShapes.end()) {
					// A cotangent: the leading symbolic dimensions of the output's shape are its `...`.
					unsigned symbolic = 0;
					while (symbolic < c->second.size() && !c->second[symbolic].isConstant())
						++symbolic;
					smaller = symbolic < ellipsisRank;
					inputRank = symbolic;
				}
				fn.inputs.push_back(makeSignature(
						pname, input.get().type, it == inputNodes.end() ? nullptr : it->second, argvals, inputRank
				));
				if (smaller)
					fn.inputs.back().ellipsisRank = static_cast<int>(inputRank);
			}
			auto rank = [&](const Node* node) {
				const auto& name = std::get<ParameterNode>(*node).name;
				return std::pair{std::find(order.begin(), order.end(), name) - order.begin(), name};
			};
			std::sort(weights.begin(), weights.end(), [&](auto* a, auto* b) { return rank(a) < rank(b); });
			for (const auto* weight : weights) {
				const auto& pn = std::get<ParameterNode>(*weight);
				fn.weights.push_back(makeSignature(pn.name, pn.type, weight, argvals, ellipsisRank));
			}
			// The exact output shapes (the `...` of an output may have fewer dimensions than that of the inputs, e.g.
			// a sequence [batch, seq] pooled to [batch]). Not for functions with packed inputs (their types are
			// imprecise); their declared types define the ABI.
			std::optional<shapes::Shapes> inferred;
			if (!function.context || function.context->producer) {
				if (auto r = shapes::infer(decl, ellipsisRank, evaluate, knownInputs(decl)))
					inferred = std::move(*r);
				else
					log::getLogger("compiler")->warn("{}.{}: {}", moduleName, funcName, r.error());
			}
			for (size_t i = 0; i < decl.outputs.size(); ++i) {
				const auto& out = decl.outputs[i].get();
				auto name = i < decl.outputNames.size() ? decl.outputNames[i]
						  : decl.outputs.size() == 1  ? std::string{"output"}
													  : "output" + std::to_string(i);
				const DataType type = i < decl.returnTypes.size()
											  ? std::visit([](const auto& t) -> DataType { return t; }, decl.returnTypes[i])
											  : getDataType(out);
				auto signature = makeSignature(std::move(name), type, &out, argvals, ellipsisRank);
				if (const shapes::Shape* exact = inferred ? inferred->find(out) : nullptr) {
					std::vector<int64_t> shape;
					for (const auto& d : *exact)
						shape.push_back(d.isConstant() ? d.factor : -1);
					// The declared dimensions after `...` (or all, without `...`) must match.
					const auto* declared = std::get_if<TensorType>(&type);
					if (declared != nullptr) {
						auto ellipsis = std::find_if(declared->shape.begin(), declared->shape.end(), [](const auto& d) {
							return std::holds_alternative<Elipsis>(d);
						});
						const size_t trailing = static_cast<size_t>(declared->shape.end() - ellipsis) - (ellipsis != declared->shape.end());
						const size_t fixed = ellipsis == declared->shape.end() ? declared->shape.size() : trailing;
						const auto& expected = signature.shape;
						bool ok = ellipsis == declared->shape.end() ? shape.size() == expected.size() : shape.size() >= fixed;
						for (size_t k = 1; ok && k <= fixed; ++k) {
							const int64_t want = expected[expected.size() - k], have = shape[shape.size() - k];
							ok = want < 0 || have < 0 || want == have;
						}
						if (!ok)
							throw std::runtime_error(std::format(
									"{}.{}: output {} has the shape [{}], which does not match its declared type", moduleName,
									funcName, i, [&] {
										std::string str;
										for (const auto& d : *exact)
											str += (str.empty() ? "" : ", ") + d.str();
										return str;
									}()
							));
					}
					signature.shape = std::move(shape);
				}
				fn.outputs.push_back(std::move(signature));
			}
			for (size_t i = function.firstGradient, k = 0; i < fn.outputs.size() && k < function.gradientOf.size(); ++i, ++k) {
				fn.outputs[i].accumulate = gradients.accumulate;
				fn.outputs[i].gradientOf = function.gradientOf[k];
			}
			if (function.context) {
				auto& packed = function.context->producer ? fn.outputs : fn.inputs;
				const size_t first = function.context->producer ? packed.size() - function.numPacked : 0;
				for (size_t i = first; i < first + function.numPacked; ++i)
					packed[i].packed = true;
			}
			exports.push_back(std::move(fn));
		}
	}
	for (const auto& name : gradients.functions)
		if (!differentiated.contains(name))
			throw std::runtime_error(std::format("no compiled module has a member function '{}' to differentiate", name));
	return exports;
}
