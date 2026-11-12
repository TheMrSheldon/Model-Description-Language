#ifndef MDL_PARSER_COMPILATIONUNIT_HPP
#define MDL_PARSER_COMPILATIONUNIT_HPP

#include "../computegraphv2/datatype.hpp"
#include "../computegraphv2/graph.hpp"
#include "../computegraphv2/object.hpp"

#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace mdl {
	namespace parser {
		/** @brief The state of the frontend (parse trees, definitions); needed to instantiate module templates. */
		struct FrontendState;
	} // namespace parser

	/**
     * @brief Represents a compilation unit from the parser.
     */
	struct CompilationUnit {
		using Member = std::variant<ModuleType, AggregateType, FunctionDecl>;
		/**
		 * @brief The definitions of the parsed file (not of imported files).
		 * @details Modules with hyperparameters are listed with their (uninstantiated) definition; see instantiate().
		 */
		std::map<std::string, Member> definitions;
		/** @brief All source files that were translated (the main file and all imported files). */
		std::vector<std::filesystem::path> sources;

		/** @brief The computegraph that owns all nodes. */
		Computegraph& graph() const;

		/**
		 * @brief Expands the module @p module with the given hyperparameter values (by hyperparameter name).
		 * @details Values are ConstNodes for scalar hyperparameters and AggregateInstances for struct-typed ones.
		 * Instances are cached: instantiating a module twice with the same values yields the same members.
		 * @return the instantiated module type (valid as long as the compilation unit) or an error message (errors in the
		 * module body are also reported).
		 */
		std::expected<std::reference_wrapper<const ModuleType>, std::string>
		instantiate(const std::string& module, const std::map<std::string, Object>& values) const;

		/** @brief Stores a derived function (e.g. a gradient) for the lifetime of the compilation unit. */
		const FunctionDecl& keep(FunctionDecl decl) const;

		std::shared_ptr<parser::FrontendState> state;
	};
} // namespace mdl

#endif
