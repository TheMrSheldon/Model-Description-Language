// textDocument/signatureHelp, and the callee and parameter helpers shared with the inlay hints.
#include "./features.hpp"

namespace {
	using mdl::analysis::SymbolKind;

	/** @brief The definition of @p name visible at @p position (innermost first). */
	const mdl::analysis::Symbol*
	visible(const mdls::Document& doc, const std::string& name, mdl::SourcePosition position) {
		for (const auto* s : doc.index->visibleAt(doc.sourceName(), doc.toIndex(position)))
			if (s->name == name)
				return s;
		return nullptr;
	}

	/** @brief The types of the value of the identifier token @p t (from its reference, else its definition's). */
	std::set<std::string> typesOf(const mdls::Document& doc, const mdls::Token& t) {
		if (const auto* r = doc.index->referenceAt(doc.sourceName(), doc.toIndex(t.begin)))
			if (!r->types.empty() && r->range.end.column - r->range.begin.column == t.text.size())
				return r->types;
		if (const auto* s = visible(doc, t.text, t.begin)) {
			const size_t id = static_cast<size_t>(s - doc.index->symbols().data());
			for (const auto& r : doc.index->references())
				if (r.symbol == id && !r.types.empty())
					return r.types;
		}
		return {};
	}

	/** @brief The member function @p name of a module that one of @p types names. */
	const mdl::analysis::Symbol*
	method(const mdls::Document& doc, const std::set<std::string>& types, const std::string& name) {
		for (const auto& s : doc.index->symbols())
			if (s.kind == SymbolKind::method && s.name == name && types.contains(s.container))
				return &s;
		return nullptr;
	}
} // namespace

mdls::Callee mdls::calleeAt(const Document& doc, const std::vector<Token>& tokens, size_t open) {
	if (doc.index == nullptr || open == 0 || !tokens[open - 1].identifier())
		return {};
	const Token& name = tokens[open - 1];
	if (tokens[open].is("[")) {
		const auto* s = visible(doc, name.text, name.begin);
		return s != nullptr && s->kind == SymbolKind::module ? Callee{s, true} : Callee{};
	}
	if (open >= 3 && tokens[open - 2].is(".") && tokens[open - 3].identifier()) // inst.f(
		return {method(doc, typesOf(doc, tokens[open - 3]), name.text), false};
	const auto* s = visible(doc, name.text, name.begin);
	if (s == nullptr)
		return {};
	if (s->kind == SymbolKind::function || s->kind == SymbolKind::method)
		return {s, false};
	return {method(doc, typesOf(doc, name), "forward"), false}; // an instance: its forward
}

std::vector<mdls::ParameterSpan> mdls::parameterSpans(const mdl::analysis::Symbol& callee, char bracket) {
	const std::string& d = callee.detail;
	const char close = bracket == '(' ? ')' : ']';
	size_t start = d.find(callee.name);
	start = start == std::string::npos ? std::string::npos : d.find(bracket, start + callee.name.size());
	std::vector<ParameterSpan> spans;
	if (start == std::string::npos)
		return spans;
	int depth = 0;
	size_t begin = start + 1;
	auto add = [&](size_t end) {
		while (begin < end && d[begin] == ' ')
			++begin;
		if (begin < end) {
			size_t n = begin;
			while (n < end && (std::isalnum(static_cast<unsigned char>(d[n])) || d[n] == '_'))
				++n;
			spans.push_back({begin, end, d.substr(begin, n - begin)});
		}
	};
	for (size_t i = start + 1; i < d.size(); ++i) {
		const char c = d[i];
		if (c == '(' || c == '[')
			++depth;
		else if ((c == ')' || c == ']') && depth > 0)
			--depth;
		else if (c == close && depth == 0) {
			add(i);
			break;
		} else if (c == ',' && depth == 0) {
			add(i);
			begin = i + 1;
		}
	}
	return spans;
}

std::string
mdls::parameterDocumentation(const Document& doc, const mdl::analysis::Symbol& callee, const std::string& name) {
	for (const auto& s : doc.index->symbols()) {
		if (s.name != name || s.file != callee.file)
			continue;
		const bool hyperparameter = s.kind == SymbolKind::hyperparameter && s.container == callee.name;
		const bool parameter = (s.kind == SymbolKind::parameter || s.kind == SymbolKind::templateParameter) &&
							   s.scope.contains(callee.range.begin);
		if (hyperparameter || parameter)
			return s.documentation;
	}
	return {};
}

mdls::json mdls::signatureHelp(const Document& doc, const json& params) {
	if (doc.index == nullptr)
		return nullptr;
	const auto cursor = toSource(LineIndex{doc.text}, params.at("position"));
	const auto tokens = tokenize(doc.text);
	// The innermost open bracket before the cursor and the argument the cursor is in.
	std::vector<size_t> before;
	for (size_t i = 0; i < tokens.size() && tokens[i].begin < cursor; ++i)
		if (!tokens[i].comment())
			before.push_back(i);
	int depth = 0;
	size_t commas = 0, open = 0;
	bool found = false;
	for (size_t k = before.size(); k-- > 0;) {
		const Token& t = tokens[before[k]];
		if (t.is(")") || t.is("]"))
			++depth;
		else if ((t.is("(") || t.is("[")) && depth-- == 0) {
			open = before[k];
			found = true;
			break;
		} else if (t.is(",") && depth == 0)
			++commas;
		else if ((t.is(";") || t.is("{") || t.is("}")) && depth == 0)
			break;
	}
	if (!found)
		return nullptr;
	const auto callee = calleeAt(doc, tokens, open);
	if (callee.symbol == nullptr)
		return nullptr;
	const auto spans = parameterSpans(*callee.symbol, callee.hyperparameters ? '[' : '(');
	json parameters = json::array();
	for (const auto& span : spans) {
		json p{{"label", json::array({span.begin, span.end})}};
		if (auto text = parameterDocumentation(doc, *callee.symbol, span.name); !text.empty())
			p["documentation"] = {{"kind", "markdown"}, {"value", text}};
		parameters.push_back(std::move(p));
	}
	// The active parameter: by position, or by name for `.name=` (hyperparameters).
	size_t active = commas;
	for (size_t k = before.size(); k-- > 0 && before[k] > open;)
		if (tokens[before[k]].is(".") && before[k] + 1 < tokens.size() && tokens[before[k] + 1].identifier() &&
			(k == 0 || tokens[before[k - 1]].is(",") || tokens[before[k - 1]].is("["))) {
			for (size_t i = 0; i < spans.size(); ++i)
				if (spans[i].name == tokens[before[k] + 1].text)
					active = i;
			break;
		} else if (tokens[before[k]].is(",")) {
			break;
		}
	json signature{{"label", callee.symbol->detail}, {"parameters", parameters}};
	if (!callee.symbol->documentation.empty())
		signature["documentation"] = {{"kind", "markdown"}, {"value", callee.symbol->documentation}};
	return {{"signatures", json::array({signature})}, {"activeSignature", 0}, {"activeParameter", active}};
}
