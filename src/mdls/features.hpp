#ifndef MDLS_FEATURES_HPP
#define MDLS_FEATURES_HPP

#include "./document.hpp"
#include "./tokens.hpp"

#include <mdl/format.hpp>
#include <mdl/parser.hpp>

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

/** @brief The requests of the language server, one file each; the server dispatches to them. */
namespace mdls {
	/** @brief A request that failed (sent as a JSON-RPC error). */
	struct RequestError : std::runtime_error {
		int code;
		RequestError(int code, const std::string& message) : std::runtime_error(message), code(code) {}
	};

	/** @brief JSON-RPC error codes. */
	enum ErrorCode : int {
		ParseError = -32700,
		MethodNotFound = -32601,
		InvalidParams = -32602,
		InternalError = -32603,
		RequestFailed = -32803,
	};

	/**
	 * @brief The callable before an opening bracket at token @p open of @p tokens: a function (`f(`), a member function
	 * of an instance (`inst.f(`), an instance's forward (`inst(`), or a module's hyperparameters (`Module[`).
	 * @return the definition and whether its parameters are the hyperparameters (between `[]`), if known.
	 */
	struct Callee {
		const mdl::analysis::Symbol* symbol = nullptr;
		bool hyperparameters = false;
	};
	Callee calleeAt(const Document& doc, const std::vector<Token>& tokens, size_t open);

	/** @brief The parameters of a declaration as written (between the first @p bracket after its name and the match). */
	struct ParameterSpan {
		size_t begin, end; ///< offsets in the declaration
		std::string name;
	};
	std::vector<ParameterSpan> parameterSpans(const mdl::analysis::Symbol& callee, char bracket);
	/** @brief The documentation of the parameter @p name of @p callee (its `@param` text), if any. */
	std::string
	parameterDocumentation(const Document& doc, const mdl::analysis::Symbol& callee, const std::string& name);

	json hover(const Document& doc, const json& params);
	json signatureHelp(const Document& doc, const json& params);
	json inlayHints(const Document& doc, const json& params);
	json references(const Document& doc, const json& params, const std::map<std::string, std::string>& openTexts);
	json documentHighlight(const Document& doc, const json& params);
	json prepareRename(const Document& doc, const json& params);
	json rename(const Document& doc, const json& params, const std::map<std::string, std::string>& openTexts);
	json foldingRanges(const Document& doc);
	/**
	 * @param stdNames the module of each definition of the standard library (e.g. `softmax` -> `std.activations`),
	 * for the quick fix that adds an import
	 */
	json codeActions(
			const Document& doc, const json& params, const std::map<std::string, std::string>& openTexts,
			const std::map<std::string, std::string>& stdNames
	);
	json
	rangeFormatting(const Document& doc, const json& params, const std::optional<mdl::format::FormatOptions>& style);
	json definition(const Document& doc, const json& params);
	json documentSymbols(const Document& doc);
	json completion(const Document& doc, const json& params, const mdl::parser::ParseOptions& options);
	/** @param style from the document's `.mdl-format`, if any (else the editor's options apply) */
	json formatting(const Document& doc, const json& params, const std::optional<mdl::format::FormatOptions>& style);
} // namespace mdls

#endif
