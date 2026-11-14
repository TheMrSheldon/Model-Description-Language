#ifndef MDLS_FEATURES_HPP
#define MDLS_FEATURES_HPP

#include "./document.hpp"

#include <mdl/format.hpp>
#include <mdl/parser.hpp>

#include <stdexcept>
#include <string>

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

	json hover(const Document& doc, const json& params);
	json definition(const Document& doc, const json& params);
	json documentSymbols(const Document& doc);
	json completion(const Document& doc, const json& params, const mdl::parser::ParseOptions& options);
	/** @param style from the document's `.mdl-format`, if any (else the editor's options apply) */
	json formatting(const Document& doc, const json& params, const std::optional<mdl::format::FormatOptions>& style);
} // namespace mdls

#endif
