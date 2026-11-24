// Syntax highlighting of code in languages that Doxygen does not highlight itself (CMake, Bash, YAML, JSON, Lua, ...),
// with highlight.js (loaded in header.html). Doxygen drops the language of fenced code blocks, so a block is marked by a
// surrounding <div class="lang-NAME"> (see docs/dev/technical/documentation.md). The colors come from doxygen-awesome's
// fragment colors (mdl.css), so they follow the light and dark theme.
(function () {
	/** Splits highlighted HTML into lines, closing the open <span>s at each line end and reopening them on the next. */
	function splitLines(html) {
		const lines = [];
		let open = [], line = "";
		for (const token of html.split(/(<span[^>]*>|<\/span>|\n)/)) {
			if (token === "\n") {
				lines.push(line + "</span>".repeat(open.length));
				line = open.join("");
			} else {
				if (token.startsWith("<span")) open.push(token);
				else if (token === "</span>") open.pop();
				line += token;
			}
		}
		lines.push(line);
		return lines;
	}

	document.addEventListener("DOMContentLoaded", () => {
		if (typeof hljs === "undefined") return; // offline: the code stays plain
		for (const block of document.querySelectorAll("div[class^='lang-'], div[class*=' lang-']")) {
			const language = [...block.classList].find((c) => c.startsWith("lang-")).slice(5);
			if (!hljs.getLanguage(language)) continue;
			for (const fragment of block.querySelectorAll(".fragment")) {
				const lines = [...fragment.querySelectorAll(".line")];
				const code = lines.map((l) => l.textContent).join("\n");
				const highlighted = splitLines(hljs.highlight(code, { language, ignoreIllegals: true }).value);
				lines.forEach((l, i) => (l.innerHTML = highlighted[i] ?? ""));
			}
		}
	});
})();
