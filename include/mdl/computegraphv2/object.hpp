#ifndef MDL_COMPUTEGRAPH_OBJECT_HPP
#define MDL_COMPUTEGRAPH_OBJECT_HPP

#include "./function.hpp"
#include "./node_decl.hpp"

#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mdl {
	struct AggregateInstance;

	/**
	 * @brief A first-class value in the MDL language.
	 * @details Either a compute-graph node (const Node*), a function definition
	 *          (FunctionDecl), or a struct instance (AggregateInstance).
	 */
	using Object = std::variant<const Node*, FunctionDecl, AggregateInstance>;

	/**
	 * @brief Represents an instance of an AggregateType (struct) at compile time.
	 * @details Holds one Object per named member. The member values are stored as
	 *          std::any (each holding an Object) to break the otherwise-circular
	 *          AggregateInstance → Object → AggregateInstance type dependency.
	 */
	class AggregateInstance {
	private:
		AggregateType type;
		std::vector<Object> members;

		AggregateInstance(AggregateType type, std::vector<Object> members)
				: type(std::move(type)), members(std::move(members)) {}

	public:
		static std::expected<AggregateInstance, std::string>
		create(AggregateType type, std::map<std::string, Object> values);

		std::optional<std::reference_wrapper<const Object>> getMember(std::string_view name) const;
	};

	std::optional<std::reference_wrapper<const FunctionDecl>>
	getMember(const Object& obj, std::string_view name, const std::vector<DataType>& argtypes);
}; // namespace mdl

#endif
