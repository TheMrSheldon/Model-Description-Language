#include <mdl/utils/pathpattern.hpp>

#include <cstddef>
#include <utility>

using mdl::PathPatterns;

namespace {
	std::vector<std::string> splitSegments(std::string_view path) {
		std::vector<std::string> segments;
		while (true) {
			auto dot = path.find('.');
			segments.emplace_back(path.substr(0, dot));
			if (dot == std::string_view::npos)
				return segments;
			path = path.substr(dot + 1);
		}
	}

	std::string_view trim(std::string_view s) {
		while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r'))
			s.remove_prefix(1);
		while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
			s.remove_suffix(1);
		return s;
	}

	/** @brief Glob match of a single segment (`*` and `?`). */
	bool globMatch(std::string_view pattern, std::string_view text) {
		size_t p = 0, t = 0, starP = std::string_view::npos, starT = 0;
		while (t < text.size()) {
			if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
				++p;
				++t;
			} else if (p < pattern.size() && pattern[p] == '*') {
				starP = p++;
				starT = t;
			} else if (starP != std::string_view::npos) {
				p = starP + 1;
				t = ++starT;
			} else {
				return false;
			}
		}
		while (p < pattern.size() && pattern[p] == '*')
			++p;
		return p == pattern.size();
	}

	/**
	 * @brief Whether the pattern segments [pi, end) match a prefix of the path segments [si, end) (a match of a
	 * prefix selects the whole subtree).
	 */
	bool matchesPrefix(
			const std::vector<std::string>& pattern, size_t pi, const std::vector<std::string>& path, size_t si
	) {
		if (pi == pattern.size())
			return true; // the pattern is exhausted: matched a prefix (or the whole path)
		if (pattern[pi] == "**") {
			for (size_t skip = si; skip <= path.size(); ++skip)
				if (matchesPrefix(pattern, pi + 1, path, skip))
					return true;
			return false;
		}
		return si < path.size() && globMatch(pattern[pi], path[si]) && matchesPrefix(pattern, pi + 1, path, si + 1);
	}
} // namespace

PathPatterns::PathPatterns(std::vector<std::string> lines) {
	for (auto& line : lines) {
		std::string_view text = trim(line);
		if (text.empty() || text.front() == '#')
			continue;
		bool negated = text.front() == '!';
		if (negated)
			text.remove_prefix(1);
		patterns.push_back(Pattern{.segments = splitSegments(trim(text)), .negated = negated});
	}
}

PathPatterns PathPatterns::parse(std::string_view text) {
	std::vector<std::string> lines;
	while (!text.empty()) {
		auto nl = text.find('\n');
		lines.emplace_back(text.substr(0, nl));
		if (nl == std::string_view::npos)
			break;
		text = text.substr(nl + 1);
	}
	return PathPatterns{std::move(lines)};
}

bool PathPatterns::selects(std::string_view path) const {
	bool selected = patterns.empty() || patterns.front().negated;
	const auto segments = splitSegments(path);
	for (const auto& pattern : patterns)
		if (matchesPrefix(pattern.segments, 0, segments, 0))
			selected = !pattern.negated;
	return selected;
}
