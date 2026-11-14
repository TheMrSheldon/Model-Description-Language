#ifndef MDL_ANALYSIS_INDEX_HPP
#define MDL_ANALYSIS_INDEX_HPP

#include "../diagnostics.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

/** @brief What tools (e.g. the language server) learn about a program besides its diagnostics. */
namespace mdl::analysis {
	enum class SymbolKind {
		module,
		structure,
		function,
		method,			   ///< a member function of a module
		hyperparameter,	   ///< of a module
		templateParameter, ///< of a function (`template<N: uint>`)
		member,			   ///< a weight, submodule, or struct member
		parameter,		   ///< of a function
		variable,		   ///< a local variable or slotted tensor
		loopVariable,	   ///< of `for` or of an array of modules
	};

	/** @brief A definition in a source file. */
	struct Symbol {
		std::string name;
		SymbolKind kind;
		std::string file;
		SourceRange range;			 ///< of the name
		std::string container;		 ///< the enclosing module or struct (empty for global definitions)
		std::string detail;			 ///< the declaration as written, e.g. `W: float[outdim, indim]`
		std::string documentation;	 ///< from the doc comment before it (Markdown)
		SourceRange scope;			 ///< where it is visible (for globals: the whole file)
		std::set<std::string> types; ///< the types it has (one per distinct elaboration, e.g. symbolic and concrete)
	};

	/** @brief A use of a name (or a declaration's name) with what is known there. */
	struct Reference {
		std::string file;
		SourceRange range;
		std::optional<size_t> symbol; ///< index into SemanticIndex::symbols, if the definition is known
		std::set<std::string> types;  ///< the type of the value there (one per distinct elaboration)
		/** @brief The members that `x.` can access after this name (module members, struct members, `data`/`mask`). */
		std::set<std::string> members;
	};

	/** @brief Definitions and references of a program (see parser::analyze), from all its elaborations. */
	class SemanticIndex {
	public:
		/** @brief Records a definition (once per position); returns its index. */
		size_t define(Symbol symbol);
		/** @brief Records a use (merged with earlier records at the same position). */
		void reference(Reference reference);

		/** @brief The definition whose name starts at @p position of @p file, if recorded. */
		std::optional<size_t> find(const std::string& file, SourcePosition position) const {
			auto it = symbolAt_.find(std::tuple{file, position.line, position.column});
			return it == symbolAt_.end() ? std::nullopt : std::optional<size_t>{it->second};
		}

		const std::vector<Symbol>& symbols() const { return symbols_; }
		const std::vector<Reference>& references() const { return references_; }
		Symbol& symbol(size_t index) { return symbols_[index]; }

		/** @brief The reference (or declaration name) at @p position of @p file, if any. */
		const Reference* referenceAt(std::string_view file, SourcePosition position) const;
		/** @brief The definitions visible at @p position of @p file (innermost first; globals of all files last). */
		std::vector<const Symbol*> visibleAt(std::string_view file, SourcePosition position) const;

	private:
		std::vector<Symbol> symbols_;
		std::vector<Reference> references_;
		std::map<std::tuple<std::string, size_t, size_t>, size_t> symbolAt_, referenceAt_;
	};

	/**
	 * @brief The Markdown of a Doxygen-style doc comment (`/** ... *\/` or `///` lines): `@brief`, `@details`,
	 * `@param`, `@return`, `@todo`, `\p`, `\f$...\f$`, `\f[...\f]`; other commands are dropped.
	 */
	std::string docCommentToMarkdown(std::string_view comment);
} // namespace mdl::analysis

#endif
