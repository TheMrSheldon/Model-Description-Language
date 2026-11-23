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

		AggregateInstance(AggregateType type, std::vector<Object> members);

	public:
		// Defined in object.cpp, where all alternatives of Object are complete (libc++ checks them when the members
		// of std::vector<Object> are instantiated).
		AggregateInstance(const AggregateInstance&);
		AggregateInstance(AggregateInstance&&) noexcept;
		AggregateInstance& operator=(const AggregateInstance&);
		AggregateInstance& operator=(AggregateInstance&&) noexcept;
		~AggregateInstance();

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

		ModuleInstance(BoundModuleType type, std::vector<Object> params);

	public:
		// Defined in object.cpp (see AggregateInstance).
		ModuleInstance(const ModuleInstance&);
		ModuleInstance(ModuleInstance&&) noexcept;
		ModuleInstance& operator=(const ModuleInstance&);
		ModuleInstance& operator=(ModuleInstance&&) noexcept;
		~ModuleInstance();

		static std::expected<ModuleInstance, std::string>
		create(BoundModuleType type, std::map<std::string, Object> values);

		std::optional<std::reference_wrapper<const Object>> getMember(std::string_view name) const;
	};

	/**
	 * @brief An array of objects, e.g., the module instances of `layers: [Encoder[conf] for i in 0..12]`.
	 */
	struct ArrayInstance {
		std::vector<Object> elements;
		/**
		 * @brief Whether the only element stands for all elements of an array of unknown size (symbolic analysis of a
		 * module without hyperparameter values): every index selects it.
		 */
		bool representative = false;
	};

	std::optional<std::reference_wrapper<const FunctionDecl>>
	getMember(const Object& obj, std::string_view name, const std::vector<DataType>& argtypes);
}; // namespace mdl

#endif
