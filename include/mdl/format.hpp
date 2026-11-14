#ifndef MDL_FORMAT_HPP
#define MDL_FORMAT_HPP

#include "./diagnostics.hpp"

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief The formatter of `.dlm` files (mdl-format, and the language server's formatting).
 * @details It changes only whitespace: indentation, spaces between tokens, line breaks, and blank lines; comments are
 * kept (re-indented). The result is checked to contain the same tokens as the input.
 */
namespace mdl::format {
	/** @brief Naming conventions of identifiers. */
	enum class Naming {
		any,		///< not checked
		snakeCase,	///< `snake_case`
		camelCase,	///< `camelCase`
		pascalCase, ///< `PascalCase`
		allCaps,	///< `ALL_CAPS` (upper snake case)
		lowercase,	///< `lowercase` (no separators)
	};

	bool isSnakeCase(std::string_view name);
	bool isCamelCase(std::string_view name);
	bool isPascalCase(std::string_view name);
	bool isAllCaps(std::string_view name);
	bool isLowercase(std::string_view name);
	/** @brief Whether @p name follows @p naming (always true for Naming::any). */
	bool follows(std::string_view name, Naming naming);
	/** @brief @p name converted to @p naming (words split at `_` and case changes), e.g. `hiddenSize` to `hidden_size`. */
	std::string convert(std::string_view name, Naming naming);
	/** @brief The name of @p naming in style files (`snake_case`, `camelCase`, `PascalCase`, `ALL_CAPS`, `lowercase`, `any`). */
	std::string_view namingName(Naming naming);

	/** @brief The naming convention per kind of definition (only definitions are checked, not uses). */
	struct NamingOptions {
		Naming module = Naming::any;
		Naming structure = Naming::any;
		Naming structMember = Naming::any;
		Naming moduleMember = Naming::any; ///< weights and submodules
		Naming function = Naming::any;	   ///< free and member functions
		Naming parameter = Naming::any;
		Naming hyperparameter = Naming::any;
		Naming templateParameter = Naming::any;
		Naming variable = Naming::any; ///< local and loop variables
	};

	/** @brief The style options (file `.mdl-format`, see loadStyle()). */
	struct FormatOptions {
		unsigned indentWidth = 4;	 ///< spaces per indentation level (continuation lines get twice as many)
		unsigned columnLimit = 120;	 ///< longer lines are wrapped (after commas, else before binary operators)
		unsigned maxEmptyLines = 1;	 ///< blank lines kept between definitions and statements
		bool reflowComments = false; ///< comment lines longer than the column limit are broken between words
		NamingOptions naming;
	};

	struct FormatResult {
		std::string text; ///< the formatted source (the input if it cannot be formatted)
		/** @brief Syntax errors (a file with syntax errors is not formatted) and naming warnings (flag `naming`). */
		std::vector<Diagnostic> diagnostics;
		bool ok = false;
	};

	/** @brief Formats the source code @p source (named @p sourcename in diagnostics). */
	FormatResult
	format(std::string_view source, const FormatOptions& options = {}, std::string_view sourcename = "<stdin>");

	/**
	 * @brief The naming warnings of @p source (empty if it has syntax errors); format() reports them as well.
	 */
	std::vector<Diagnostic>
	checkNaming(std::string_view source, const NamingOptions& naming, std::string_view sourcename = "<stdin>");

	/**
	 * @brief Reads a style file: `Key: value` lines (a flat YAML map; `#` starts a comment). Keys: `IndentWidth`,
	 * `ColumnLimit`, `MaxEmptyLines`, `ReflowComments` (true/false), and the naming conventions `ModuleNaming`,
	 * `StructNaming`, `StructMemberNaming`, `ModuleMemberNaming`, `FunctionNaming`, `ParameterNaming`,
	 * `HyperparameterNaming`, `TemplateParameterNaming`, `VariableNaming` (see namingName()). Unknown keys and invalid
	 * values are errors.
	 */
	std::expected<FormatOptions, std::string> loadStyle(const std::filesystem::path& file);

	/** @brief The style file `.mdl-format` in the directory of @p source or the closest parent directory, if any. */
	std::optional<std::filesystem::path> findStyleFile(const std::filesystem::path& source);
} // namespace mdl::format

#endif
