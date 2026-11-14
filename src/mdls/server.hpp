#ifndef MDLS_SERVER_HPP
#define MDLS_SERVER_HPP

#include "./protocol.hpp"

#include <mdl/parser.hpp>

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace mdls {
	/**
	 * @brief The language server: reads requests and notifications from @p in, writes responses and notifications to
	 * @p out. Documents are analyzed when they are opened or changed (the full text is synchronized).
	 */
	class Server {
	public:
		Server(std::istream& in, std::ostream& out, std::ostream* log = nullptr) : in(in), out(out), log(log) {}

		/** @brief Serves until `exit` (or the end of the input); returns the process exit code. */
		int run();

	private:
		struct Document {
			std::filesystem::path path;
			std::string text;
			int version = 0;
			/** @brief The index of the last analysis that could translate the document (kept while it has syntax errors). */
			std::shared_ptr<const mdl::analysis::SemanticIndex> index;
			std::string indexText; ///< the text that @p index is from

			/**
			 * @brief The position in indexText of @p position in text: lines before and after the changed lines map
			 * to themselves, positions in changed lines to the start of the change.
			 */
			mdl::SourcePosition toIndex(mdl::SourcePosition position) const;
		};

		std::istream& in;
		std::ostream& out;
		std::ostream* log;
		std::map<std::string, Document> documents; ///< by URI
		mdl::parser::ParseOptions parseOptions;	   ///< from the client's initialization options
		bool shutdownRequested = false;

		void handle(const json& message);
		json handleRequest(const std::string& method, const json& params);
		void handleNotification(const std::string& method, const json& params);

		json initialize(const json& params);
		json formatting(const json& params);
		json semanticTokens(const json& params);
		json hover(const json& params);
		json definition(const json& params);
		json completion(const json& params);
		json documentSymbols(const json& params);

		Document& document(const json& params);
		void analyze(Document& document);
		void notify(const std::string& method, json params);
		void logLine(const std::string& text);
	};
} // namespace mdls

#endif
