#ifndef MDLS_DOCUMENT_HPP
#define MDLS_DOCUMENT_HPP

#include "./protocol.hpp"

#include <mdl/analysis/index.hpp>
#include <mdl/diagnostics.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mdls {
	/** @brief An open document and what the last analysis learned about it. */
	struct Document {
		std::string uri;			///< as sent by the client (published diagnostics use it unchanged)
		std::filesystem::path path; ///< the file (for `untitled:` documents: the URI)
		bool onDisk = false;		///< whether the URI is a `file:` URI
		std::string text;
		int version = 0;
		/** @brief The files the last analysis read (the document and its imports), to re-analyze importers. */
		std::vector<std::filesystem::path> sources;
		/** @brief The index of the last analysis that could translate the document (kept while it has syntax errors). */
		std::shared_ptr<const mdl::analysis::SemanticIndex> index;
		std::string indexText; ///< the text that @p index is from

		/** @brief The name of the document in the frontend's diagnostics and index. */
		std::string sourceName() const { return path.string(); }

		/**
		 * @brief The position in indexText of @p position in text: lines before and after the changed lines map to
		 * themselves, positions in changed lines to the start of the change.
		 */
		mdl::SourcePosition toIndex(mdl::SourcePosition position) const;
		/** @brief The inverse of toIndex(): std::nullopt for positions in the changed lines. */
		std::optional<mdl::SourcePosition> fromIndex(mdl::SourcePosition position) const;
	};

	/** @brief The frontend's position (line from 1, code points) of a protocol position (line from 0, UTF-16). */
	mdl::SourcePosition toSource(const LineIndex& lines, const json& position);
	/** @brief The protocol's position and range of the frontend's. */
	json toProtocol(const LineIndex& lines, mdl::SourcePosition position);
	json toProtocol(const LineIndex& lines, const mdl::SourceRange& range);
} // namespace mdls

#endif
