#include <mdl/computegraphv2/graph.hpp>

#include <mdl/computegraphv2/function.hpp>
#include <mdl/passes/shapes.hpp>

#include <cassert>
#include <format>
#include <functional>
#include <mdl/utils/overload.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mdl;
using mdl::builtins::FunctionIDs;

Computegraph::Computegraph() = default;
Computegraph::Computegraph(Computegraph&&) = default;
Computegraph& Computegraph::operator=(Computegraph&&) = default;

namespace {
	/** @brief Appends the bytes of @p value to the structural key @p key. */
	template <typename T>
	void appendKey(std::string& key, const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		key.append(reinterpret_cast<const char*>(&value), sizeof(T));
	}
} // namespace

const Node& Computegraph::addConstNode(Atomic&& value) {
	std::string key = "c";
	appendKey(key, static_cast<char>(value.index()));
	std::visit(
			overloaded{
					[&](int v) { appendKey(key, v); },
					[&](double v) { appendKey(key, v); }, // bitwise: distinguishes -0.0 and NaN payloads
					[&](const std::vector<int>& v) {
						appendKey(key, v.size());
						for (int i : v)
							appendKey(key, i);
					},
			},
			value
	);
	auto [it, inserted] = interned.try_emplace(std::move(key), nullptr);
	if (!inserted) {
		++numReused;
		return *it->second;
	}
	it->second = &nodes.emplace_back(ConstNode{.value = std::move(value)});
	return *it->second;
}

namespace {
	/** @brief Whether the result of @p func is masked if an argument is (like torch.masked: masked in, masked out). */
	bool propagatesMask(FunctionIDs func) {
		switch (func) {
		case FunctionIDs::SUM_ALL:
		case FunctionIDs::MEAN_ALL:
		case FunctionIDs::MASK_OF:
		case FunctionIDs::UNMASK:
			return false;
		default:
			return true;
		}
	}
} // namespace

const Node& Computegraph::addInvocationNode(FunctionIDs func, std::vector<NodeRef>&& args) {
	DataType dtype = shapes::resultType(func, args);
	if (auto* tensor = std::get_if<TensorType>(&dtype); tensor != nullptr && propagatesMask(func))
		for (const auto& arg : args)
			if (const auto* t = std::get_if<TensorType>(&getDataType(arg.get())); t != nullptr && t->masked)
				tensor->masked = true;
	std::string key = "i";
	appendKey(key, func);
	appendKey(key, args.size());
	for (const auto& arg : args)
		appendKey(key, &arg.get());
	// (The type follows from the arguments.)
	if (auto it = interned.find(key); it != interned.end()) {
		++numReused;
		return *it->second;
	}

	bool constevalable = std::all_of(args.begin(), args.end(), isCompiletimeEvalable);
	const Node& node = nodes.emplace_back(
			InvocationNode{
					.type = std::move(dtype),
					.constevalable = constevalable,
					.id = func,
					.arguments = std::move(args),
			}
	);
	interned.emplace(std::move(key), &node);
	return node;
}

namespace {
	/**
	 * @brief Clones the subgraph of a function template into the graph while substituting nodes.
	 * @details Every template node is cloned at most once (memoization), so shared subexpressions of the template stay
	 * shared in the clone. Nodes that do not (transitively) depend on a substituted node are not cloned at all but
	 * reused.
	 */
	class Inliner final {
	public:
		/** Maps a weight of the template onto the weight to use in the clone (or nullptr to keep it). */
		using WeightMap = std::function<const Node*(const ParameterNode&)>;

		Inliner(Computegraph& graph, WeightMap weightMap = nullptr) : graph(graph), weightMap(std::move(weightMap)) {}

		/**
		 * @brief Substitutes a function argument or template/module hyperparameter by @p to.
		 * @details FunctionDecl and ModuleType reference the node alternatives (e.g., `const InputNode&`) rather than
		 * the Node variants, so leaf substitutions are keyed by the address of the alternative.
		 */
		template <typename Leaf>
		void substitute(const Leaf& from, const Node& to) {
			leafSubs[static_cast<const void*>(&from)] = &to;
		}

		/** @brief Replaces the node @p from (of any kind) by @p to. */
		void replace(const Node& from, const Node& to) { memo[&from] = &to; }

		const Node& clone(const Node& node) {
			if (auto it = memo.find(&node); it != memo.end())
				return *it->second;
			auto leaf = [&](const auto& alt) -> const Node& {
				auto it = leafSubs.find(static_cast<const void*>(&alt));
				return it == leafSubs.end() ? node : *it->second;
			};
			const Node& result = std::visit(
					overloaded{
							[&](const InvocationNode& invn) -> const Node& { return cloneInvocation(node, invn); },
							[&](const ParameterNode& pn) -> const Node& {
								if (weightMap)
									if (const Node* mapped = weightMap(pn))
										return *mapped;
								return node;
							},
							[&](const InputNode& in) -> const Node& { return leaf(in); },
							[&](const HyperparameterNode& hn) -> const Node& { return leaf(hn); },
							// Constants and errors are kept as they are.
							[&](const auto&) -> const Node& { return node; },
					},
					node
			);
			memo[&node] = &result;
			return result;
		}

		/**
		 * @brief Records the module calls within the inlined subgraph for the clone (see ModuleCall).
		 * @details Must be called after cloning. @p rescope adapts the weight scope and instance path of each call (e.g.
		 * prefixes the instance name of an inlined module).
		 */
		void propagateModuleCalls(const std::function<void(ModuleCall&)>& rescope) {
			std::vector<const ModuleCall*> calls;
			for (const auto& [from, to] : memo)
				for (const ModuleCall* call : graph.moduleCallsProducing(*from))
					if (std::find(calls.begin(), calls.end(), call) == calls.end())
						calls.push_back(call);
			for (const ModuleCall* call : calls) {
				ModuleCall copy = *call;
				for (auto* nodes : {&copy.inputs, &copy.outputs})
					for (auto& node : *nodes)
						node = &clone(*node);
				for (auto& [name, value] : copy.hyperparameters)
					value = &clone(*value);
				if (rescope)
					rescope(copy);
				graph.addModuleCall(std::move(copy));
			}
		}

		/** @brief Substitutes nodes that are referenced by symbolic dimensions of a (tensor) type. */
		DataType cloneType(const DataType& dtype) {
			const auto* tt = std::get_if<TensorType>(&dtype);
			if (tt == nullptr)
				return dtype;
			TensorType result{tt->type, {}};
			result.shape.reserve(tt->shape.size());
			for (const auto& dim : tt->shape) {
				if (const auto* nodePtr = std::get_if<const Node*>(&dim))
					result.shape.emplace_back(&clone(**nodePtr));
				else
					result.shape.push_back(dim);
			}
			return result;
		}

	private:
		Computegraph& graph;
		WeightMap weightMap;
		std::unordered_map<const Node*, const Node*> memo;
		std::unordered_map<const void*, const Node*> leafSubs;

		const Node& cloneInvocation(const Node& node, const InvocationNode& invn) {
			std::vector<Computegraph::NodeRef> newArgs;
			newArgs.reserve(invn.arguments.size());
			bool changed = false;
			for (const auto& arg : invn.arguments) {
				const Node& newArg = clone(arg.get());
				changed |= (&newArg != &arg.get());
				newArgs.push_back(std::cref(newArg));
			}
			if (!changed)
				return node; // Independent of all substitutions: reuse the template node.
			return graph.addInvocationNode(invn.id, std::move(newArgs));
		}
	};

	Computegraph::Outputs cloneOutputs(Inliner& inliner, const FunctionDecl& func) {
		Computegraph::Outputs outputs;
		outputs.reserve(func.outputs.size());
		for (const auto& output : func.outputs)
			outputs.push_back(std::cref(inliner.clone(output.get())));
		return outputs;
	}

	std::expected<void, std::string> checkArity(const FunctionDecl& func, size_t numArgs) {
		if (func.parameters.size() != numArgs)
			return std::unexpected(
					std::format("'{}' expects {} argument(s) but got {}", func.name, func.parameters.size(), numArgs)
			);
		return {};
	}
} // namespace

std::expected<Computegraph::Outputs, std::string> Computegraph::addInvocationNode(
		const FunctionDecl& func, std::vector<NodeRef>&& arguments, std::vector<NodeRef>&& templateArguments
) {
	if (auto ok = checkArity(func, arguments.size()); !ok)
		return std::unexpected(ok.error());
	if (func.parameters.empty() && templateArguments.empty())
		return Outputs{func.outputs.begin(), func.outputs.end()};

	Inliner inliner{*this};
	for (size_t i = 0; i < func.parameters.size(); ++i)
		inliner.substitute(func.parameters[i].second.get(), arguments[i].get());
	for (size_t i = 0; i < std::min(func.templateParameters.size(), templateArguments.size()); ++i)
		inliner.substitute(func.templateParameters[i].second.get(), templateArguments[i].get());
	auto outputs = cloneOutputs(inliner, func);
	inliner.propagateModuleCalls(nullptr); // same weight scope: the instance paths are unchanged
	return outputs;
}

std::expected<Computegraph::Outputs, std::string> Computegraph::addModuleInvocationNode(
		const BoundModuleType& boundModule, const std::string& scope, const std::string& instanceName,
		const FunctionDecl& func, std::vector<NodeRef>&& arguments
) {
	if (auto ok = checkArity(func, arguments.size()); !ok)
		return std::unexpected(ok.error());

	// Every weight referenced by a member function of the module is a weight of the instance: map `w` onto the
	// caller-scope weight `instanceName.w`. Its type may depend on the module's hyperparameters, which are
	// substituted by the bound values.
	Inliner* self = nullptr;
	Inliner inliner{*this, [&](const ParameterNode& weight) -> const Node* {
						const std::string path = instanceName.empty() ? weight.name : instanceName + "." + weight.name;
						return &getOrAddParameterNode(scope, path, self->cloneType(weight.type));
					}};
	self = &inliner;

	const auto& mod = boundModule.definition;
	for (const auto& [name, hn] : mod.typeargs)
		if (auto it = boundModule.hyperparamValues.find(name); it != boundModule.hyperparamValues.end())
			inliner.substitute(hn.get(), *it->second);
	for (size_t i = 0; i < func.parameters.size(); ++i)
		inliner.substitute(func.parameters[i].second.get(), arguments[i].get());
	auto outputs = cloneOutputs(inliner, func);

	// Module calls within the callee now belong to the instance: `inner` becomes `instanceName.inner` in `scope`.
	inliner.propagateModuleCalls([&](ModuleCall& call) {
		if (call.module == nullptr)
			return; // a call of a free function: no weights
		call.scope = scope;
		if (!instanceName.empty())
			call.path = call.path.empty() ? instanceName : instanceName + "." + call.path;
	});

	ModuleCall call{
			.scope = scope,
			.path = instanceName,
			.module = std::make_shared<const ModuleType>(boundModule.definition),
			.hyperparameters = boundModule.hyperparamValues,
			.function = func.name,
			.inputs = {},
			.outputs = {},
	};
	for (const auto& arg : arguments)
		call.inputs.push_back(&arg.get());
	for (const auto& output : outputs)
		call.outputs.push_back(&output.get());
	addModuleCall(std::move(call));
	return outputs;
}

Computegraph::Outputs Computegraph::substitute(
		const std::vector<NodeRef>& roots, const std::unordered_map<const Node*, const Node*>& replacements
) {
	Inliner inliner{*this};
	for (const auto& [from, to] : replacements)
		inliner.replace(*from, *to);
	Outputs result;
	result.reserve(roots.size());
	for (const auto& root : roots)
		result.push_back(std::cref(inliner.clone(root.get())));
	// Module calls are not propagated: a call whose outputs were replaced (e.g. by inputs) is no longer a unit.
	return result;
}

const Node* Computegraph::findParameterNode(const std::string& scope, const std::string& path) const {
	auto it = weights.find({scope, path});
	return it == weights.end() ? nullptr : it->second;
}

void Computegraph::addModuleCall(ModuleCall call) {
	if (call.outputs.empty())
		return;
	auto& producing = moduleCallsByOutput[call.outputs.front()];
	for (const ModuleCall* existing : producing)
		if (existing->scope == call.scope && existing->path == call.path && existing->function == call.function &&
			existing->inputs == call.inputs && existing->outputs == call.outputs)
			return;
	const ModuleCall& stored = moduleCalls.emplace_back(std::move(call));
	for (const Node* output : stored.outputs) {
		auto& list = moduleCallsByOutput[output];
		if (std::find(list.begin(), list.end(), &stored) == list.end())
			list.push_back(&stored);
	}
}

std::vector<const ModuleCall*> Computegraph::moduleCallsProducing(const Node& node) const {
	auto it = moduleCallsByOutput.find(&node);
	return it == moduleCallsByOutput.end() ? std::vector<const ModuleCall*>{} : it->second;
}

const Node& Computegraph::addErrorNode() {
	nodes.emplace_back(ErrorNode{});
	return nodes.back();
}

const Node& Computegraph::addInputNode(DataType dtype, std::string name) {
	return nodes.emplace_back(InputNode{.type = std::move(dtype), .name = std::move(name)});
}

const Node& Computegraph::addParameterNode(DataType dtype, std::string name) {
	return nodes.emplace_back(ParameterNode{.type = std::move(dtype), .constevalable = false, .name = std::move(name)});
}

const Node& Computegraph::getOrAddParameterNode(const std::string& scope, const std::string& path, DataType dtype) {
	auto [it, inserted] = weights.try_emplace({scope, path}, nullptr);
	if (inserted)
		it->second = &addParameterNode(std::move(dtype), path);
	return *it->second;
}

const Node& Computegraph::addHyperparameterNode(DataType dtype, std::string name) {
	return nodes.emplace_back(
			HyperparameterNode{.type = std::move(dtype), .constevalable = true, .name = std::move(name)}
	);
}

size_t Computegraph::getNumNodes() const { return nodes.size(); }
