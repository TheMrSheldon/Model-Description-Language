// Definitions that are never used: local variables, parameters, loop variables, module members (weights and
// submodules), and imports (see parser::ParseOptions::unused).
#include <mdl/analysis/index.hpp>

#include <algorithm>
#include <format>
#include <map>
#include <set>

std::vector<mdl::analysis::Unused>
mdl::analysis::unusedDefinitions(const SemanticIndex& index, const std::string& file) {
	std::vector<Unused> result;
	auto hint = [&](const SourceRange& range, std::string message) {
		result.push_back({.range = range, .message = std::move(message)});
	};
	// The uses of each definition (references other than its declaration).
	std::vector<size_t> uses(index.symbols().size(), 0);
	for (const auto& r : index.references())
		if (r.symbol) {
			const auto& s = index.symbols()[*r.symbol];
			if (!(r.file == s.file && r.range == s.range))
				++uses[*r.symbol];
		}
	for (size_t id = 0; id < index.symbols().size(); ++id) {
		const auto& s = index.symbols()[id];
		if (s.file != file || uses[id] > 0 || s.name.starts_with("_"))
			continue;
		switch (s.kind) {
		case SymbolKind::variable:
		case SymbolKind::loopVariable:
			hint(s.range, std::format("{} is never used", s.name));
			break;
		case SymbolKind::parameter:
			hint(s.range, std::format("The parameter {} is never used", s.name));
			break;
		case SymbolKind::member:
			if (!s.container.empty() && s.container.find('.') == std::string::npos &&
				std::any_of(index.symbols().begin(), index.symbols().end(), [&](const auto& m) {
					return m.kind == SymbolKind::module && m.name == s.container && m.file == file;
				}))
				hint(s.range, std::format("The member {} of {} is never used", s.name, s.container));
			break;
		default:
			break;
		}
	}
	// Imports whose files (and the files they import) define nothing that the document uses.
	std::map<size_t, const ImportPath*> statements; // line -> its full path (the last part)
	for (const auto& i : index.imports())
		if (i.file == file) {
			auto& last = statements[i.range.begin.line];
			if (last == nullptr || last->range.begin < i.range.begin)
				last = &i;
		}
	std::set<std::string> used; // the files of the definitions the document uses
	for (const auto& r : index.references())
		if (r.file == file && r.symbol && index.symbols()[*r.symbol].file != file)
			used.insert(index.symbols()[*r.symbol].file);
	for (const auto& [_, import] : statements) {
		std::set<std::string> reachable{import->targets.begin(), import->targets.end()};
		for (bool grew = true; grew;) {
			grew = false;
			for (const auto& i : index.imports())
				if (reachable.contains(i.file))
					for (const auto& t : i.targets)
						grew |= reachable.insert(t).second;
		}
		if (std::none_of(reachable.begin(), reachable.end(), [&](const auto& f) { return used.contains(f); }))
			hint(import->range, std::format("The import {} is never used", import->name));
	}
	std::ranges::sort(result, {}, [](const Unused& u) { return std::pair{u.range.begin.line, u.range.begin.column}; });
	return result;
}
