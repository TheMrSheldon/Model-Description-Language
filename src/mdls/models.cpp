// textDocument/codeLens (parameter counts and weight memory of modules) and mdl/moduleTree (the instance tree of a
// module with its weights, e.g. for a tree view).
#include "./features.hpp"

#include <format>
#include <map>

namespace {
	using mdl::analysis::ModuleLayout;
	using mdl::analysis::SemanticIndex;

	const ModuleLayout* layoutOf(const SemanticIndex& index, size_t module, const std::string& instance) {
		for (const auto& l : index.layouts())
			if (l.module == module && l.instance == instance)
				return &l;
		return nullptr;
	}

	/** @brief The number of values of a module instance: exact if all shapes are constant, else as a formula. */
	struct Count {
		bool exact = true;
		int64_t value = 0;
		std::vector<std::pair<std::string, int64_t>> terms; ///< (product, multiplicity), in order

		void add(const std::string& term, int64_t times = 1) {
			for (auto& [t, n] : terms)
				if (t == term) {
					n += times;
					return;
				}
			terms.emplace_back(term, times);
		}
		void add(const Count& other) {
			exact &= other.exact;
			value += other.value;
			for (const auto& [t, n] : other.terms)
				add(t, n);
		}
		std::string formula() const {
			std::string s;
			if (value != 0)
				s = std::to_string(value);
			for (const auto& [t, n] : terms)
				s += (s.empty() ? "" : " + ") + (n == 1 ? t : std::to_string(n) + " * " + t);
			return s.empty() ? "0" : s;
		}
	};

	Count countOf(const SemanticIndex& index, size_t module, const std::string& instance, int depth = 0) {
		Count c;
		const auto* layout = layoutOf(index, module, instance);
		if (layout == nullptr || depth > 32) {
			c.exact = false;
			c.add("?");
			return c;
		}
		for (const auto& w : layout->weights) {
			if (w.submodule) {
				if (w.module)
					c.add(countOf(index, *w.module, w.instance, depth + 1));
				continue;
			}
			if (w.count) {
				c.value += *w.count;
			} else {
				c.exact = false;
				c.add(w.formula);
			}
		}
		return c;
	}

	/** @brief A count for humans, e.g. `110.1 M`. */
	std::string human(double n, const char* unit) {
		const char* prefixes[] = {"", " K", " M", " G", " T"};
		size_t k = 0;
		while (n >= 1000 && k + 1 < std::size(prefixes)) {
			n /= 1000;
			++k;
		}
		return k == 0 ? std::format("{:.0f}{}{}", n, *unit ? " " : "", unit)
					  : std::format("{:.1f}{}{}", n, prefixes[k], unit);
	}

	std::string bytes(double n) {
		const char* units[] = {"B", "KB", "MB", "GB", "TB"};
		size_t k = 0;
		while (n >= 1000 && k + 1 < std::size(units)) {
			n /= 1000;
			++k;
		}
		return k == 0 ? std::format("{:.0f} B", n) : std::format("{:.1f} {}", n, units[k]);
	}

	/** @brief The lens text of an instance: its parameter count and the memory of its weights per float format. */
	std::string summary(const Count& c, const std::string& instance) {
		if (!c.exact)
			return std::format("{}: {} parameters", instance, c.formula());
		const double n = static_cast<double>(c.value);
		return std::format(
				"{}: {} parameters · {} (float32) · {} (bfloat16/float16) · {} (int8) · {} (int4)", instance,
				human(n, ""), bytes(4 * n), bytes(2 * n), bytes(n), bytes(n / 2)
		);
	}

	mdls::json
	tree(const SemanticIndex& index, size_t module, const std::string& instance, const std::string& name, int depth) {
		const auto count = countOf(index, module, instance);
		mdls::json node{
				{"name", name},
				{"type", instance},
				{"parameters", count.exact ? mdls::json(count.value) : mdls::json(count.formula())},
				{"children", mdls::json::array()},
		};
		const auto* layout = layoutOf(index, module, instance);
		if (layout == nullptr || depth > 32)
			return node;
		for (const auto& w : layout->weights) {
			if (w.submodule && w.module)
				node["children"].push_back(tree(index, *w.module, w.instance, w.path, depth + 1));
			else if (!w.submodule)
				node["children"].push_back(
						{{"name", w.path},
						 {"type", w.type},
						 {"parameters", w.count ? mdls::json(*w.count) : mdls::json(w.formula)},
						 {"weight", true}}
				);
		}
		return node;
	}
} // namespace

mdls::json mdls::codeLenses(const Document& doc) {
	json lenses = json::array();
	if (doc.index == nullptr || doc.indexText != doc.text)
		return lenses;
	const LineIndex lines{doc.text};
	const auto& index = *doc.index;
	for (size_t id = 0; id < index.symbols().size(); ++id) {
		const auto& s = index.symbols()[id];
		if (s.file != doc.sourceName() || s.kind != mdl::analysis::SymbolKind::module)
			continue;
		size_t shown = 0;
		for (const auto& l : index.layouts())
			if (l.module == id && shown++ < 4)
				lenses.push_back(
						{{"range", toProtocol(lines, s.range)},
						 {"command", {{"title", summary(countOf(index, id, l.instance), l.instance)}, {"command", ""}}}}
				);
	}
	return lenses;
}

mdls::json mdls::moduleTree(const Document& doc, const json& params) {
	json roots = json::array();
	if (doc.index == nullptr)
		return roots;
	const auto& index = *doc.index;
	const std::string only = params.value("module", "");
	for (size_t id = 0; id < index.symbols().size(); ++id) {
		const auto& s = index.symbols()[id];
		if (s.file != doc.sourceName() || s.kind != mdl::analysis::SymbolKind::module ||
			(!only.empty() && s.name != only))
			continue;
		for (const auto& l : index.layouts())
			if (l.module == id)
				roots.push_back(tree(index, id, l.instance, s.name, 0));
	}
	return roots;
}
