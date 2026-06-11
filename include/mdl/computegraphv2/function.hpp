#ifndef MDL_COMPUTEGRAPH_FUNCTIONS_HPP
#define MDL_COMPUTEGRAPH_FUNCTIONS_HPP

#include "./datatype.hpp"
#include "./graph.hpp"
#include "./node.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mdl {
	template <typename... Args>
	FunctionDecl bind(Computegraph& graph, const FunctionDecl& decl, Args&&... args) {
		constexpr size_t N = sizeof...(Args);
		// Decay the args once so we can use them as non-reference template arguments below.
		using ArgsTuple = std::tuple<std::decay_t<Args>...>;

		// Placeholder index for each arg: 0 = bound value, k > 0 = std::placeholders::_k
		std::array<int, N> phIdx = {std::is_placeholder<std::decay_t<Args>>::value...};
		int maxPh = N > 0 ? *std::max_element(phIdx.begin(), phIdx.end()) : 0;

		// Create new ParameterNodes for each placeholder slot.
		// placeholder _k at arg position i gets the DataType of decl.parameters[i].
		std::vector<std::string> newParamNames(maxPh);
		std::vector<DataType> newParamTypes(maxPh, PrimitiveType::error);
		for (size_t i = 0; i < N; ++i) {
			int k = phIdx[i];
			if (k > 0) {
				newParamNames[k - 1] = decl.parameters[i].first;
				newParamTypes[k - 1] = decl.parameters[i].second.get().datatype();
			}
		}
		std::vector<std::reference_wrapper<const Node>> newParamNodes;
		newParamNodes.reserve(maxPh);
		for (int k = 0; k < maxPh; ++k)
			newParamNodes.push_back(graph.addParameterNode(newParamTypes[k], newParamNames[k]));

		// Build the argument list for calling decl:
		// bound positions use the supplied node, placeholder positions use the new param node.
		std::vector<std::reference_wrapper<const Node>> invocationArgs;
		invocationArgs.reserve(N);
		auto argTuple = std::forward_as_tuple(std::forward<Args>(args)...);
		[&]<size_t... I>(std::index_sequence<I...>) {
			// Use ArgsTuple (decayed) instead of decltype(argTuple) to avoid
			// instantiating tuple_element on a reference-to-tuple, which is incomplete.
			(
					[&] {
						using Arg = std::tuple_element_t<I, ArgsTuple>;
						constexpr int ph = std::is_placeholder<Arg>::value;
						if constexpr (ph > 0) {
							invocationArgs.push_back(newParamNodes[ph - 1]);
						} else {
							invocationArgs.push_back(std::cref(static_cast<const Node&>(std::get<I>(argTuple))));
						}
					}(),
					...);
		}(std::make_index_sequence<N>{});

		auto outputOrErr = graph.addInvocationNode(decl, std::move(invocationArgs));
		assert(outputOrErr.has_value()); // bind() callers must supply correct arg counts
		const Node& outputNode = outputOrErr.value().get();

		// Build parameters first; FunctionDecl cannot be default-constructed because
		// output (reference_wrapper) has no default constructor.
		std::vector<std::pair<std::string, std::reference_wrapper<const ParameterNode>>> parameters;
		parameters.reserve(maxPh);
		for (int k = 0; k < maxPh; ++k) {
			const ParameterNode& pn = std::get<ParameterNode>(newParamNodes[k].get());
			parameters.emplace_back(newParamNames[k], std::cref(pn));
		}
		return FunctionDecl{
				.name = decl.name,
				.templateParameters = decl.templateParameters,
				.parameters = std::move(parameters),
				.output = outputNode,
		};
	}
} // namespace mdl

#endif