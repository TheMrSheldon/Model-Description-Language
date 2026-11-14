#ifndef MDLS_SERVER_HPP
#define MDLS_SERVER_HPP

#include "./document.hpp"

#include <mdl/format.hpp>
#include <mdl/parser.hpp>

#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>

namespace mdls {
	/**
	 * @brief The language server: reads requests and notifications from @p in, writes responses and notifications to
	 * @p out. Documents are analyzed when they are opened or changed (the full text is synchronized), together with the
	 * open documents that import them.
	 */
	class Server {
	public:
		Server(std::istream& in, std::ostream& out, std::ostream* log = nullptr) : in(in), out(out), log(log) {}

		/** @brief Serves until `exit` (or the end of the input); returns the process exit code. */
		int run();

	private:
		std::istream& in;
		std::ostream& out;
		std::ostream* log;
		std::map<std::string, Document> documents; ///< by URI
		mdl::parser::ParseOptions parseOptions;	   ///< from the client's initialization options
		bool shutdownRequested = false;

		/** @brief A loaded style file (reloaded if it changed on disk). */
		struct CachedStyle {
			std::filesystem::file_time_type modified;
			std::optional<mdl::format::FormatOptions> style; ///< std::nullopt if invalid
		};
		std::map<std::filesystem::path, CachedStyle> styles;
		/** @brief The module of each definition of the standard library (computed once, for import quick fixes). */
		std::optional<std::map<std::string, std::string>> stdNames;
		const std::map<std::string, std::string>& standardLibraryNames();
		/** @brief The texts of the open documents by path (for edits in other files). */
		std::map<std::string, std::string> openTexts() const;

		void handle(const json& message);
		json handleRequest(const std::string& method, const json& params);
		void handleNotification(const std::string& method, const json& params);
		json initialize(const json& params);

		Document& document(const json& params);
		/** @brief Analyzes @p document and the open documents that import it (directly or indirectly). */
		void analyzeWithImporters(Document& document);
		void analyze(Document& document);
		/** @brief The style of the closest `.mdl-format` of @p document, if any (and valid). */
		std::optional<mdl::format::FormatOptions> styleOf(const Document& document);
		void notify(const std::string& method, json params);
		void logLine(const std::string& text);
	};
} // namespace mdls

#endif
