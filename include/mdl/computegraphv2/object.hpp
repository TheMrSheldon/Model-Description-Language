#ifndef MDL_COMPUTEGRAPH_OBJECT_HPP
#define MDL_COMPUTEGRAPH_OBJECT_HPP

#include "./object_decl.hpp"

#include "./function.hpp"

#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mdl {
	/**
	 * @brief Represents an instance of an AggregateType (struct) at compile time.
	 * @details Holds one Object per named member, which in turn are again Objects.
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

		const AggregateType& getType() const { return type; }
		/** @brief The member values, in the (alphabetical) order of AggregateType::members. */
		const std::vector<Object>& getMembers() const { return members; }
	};

	/**
	 * @brief Represents an object that is an instance of a module. Module instance **must** have fully specified
	 *        template arguments.
	 */
	class ModuleInstance {
	private:
		BoundModuleType type;
		std::vector<Object> params;

		ModuleInstance(BoundModuleType type, std::vector<Object> params)
				: type(std::move(type)), params(std::move(params)) {}

	public:
		static std::expected<ModuleInstance, std::string>
		create(BoundModuleType type, std::map<std::string, Object> values);

		std::optional<std::reference_wrapper<const Object>> getMember(std::string_view name) const;
	};

	/**
	 * @brief An array of objects, e.g., the module instances of `layers: [Encoder[conf] for i in 0..12]`.
	 */
	struct ArrayInstance {
		std::vector<Object> elements;
	};

	std::optional<std::reference_wrapper<const FunctionDecl>>
	getMember(const Object& obj, std::string_view name, const std::vector<DataType>& argtypes);
}; // namespace mdl

#endif
