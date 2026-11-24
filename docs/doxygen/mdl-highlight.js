// Syntax highlighting of MDL code in the documentation. Doxygen does not know MDL and drops the language of fenced code
// blocks, so MDL code is marked by a surrounding <div class="mdl"> (mdl-doc emits it; hand-written pages can use it).
// The keywords are those of src/mdl/mdl.g4; the classes are Doxygen's own (styled by doxygen-awesome).
(function () {
	const rules = [
		[/^\/\/.*|^\/\*[\s\S]*?\*\//, "comment"],
		[/^(module|struct|template|import|from|as|public|private|return|for|in|auto|masked|array)\b/, "keyword"],
		[/^(float|bfloat16|float16|float32|int|uint|bool)\b/, "keywordtype"],
		[/^(true|false)\b/, "keyword"],
		[/^@\w+/, "preprocessor"],
		[/^(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?/, "stringliteral"],
	];
	const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

	function highlight(text) {
		let html = "";
		while (text.length > 0) {
			let matched = false;
			for (const [pattern, cls] of rules) {
				const m = text.match(pattern);
				if (m) {
					html += `<span class="${cls}">${escape(m[0])}</span>`;
					text = text.slice(m[0].length);
					matched = true;
					break;
				}
			}
			if (!matched) {
				const word = text.match(/^\w+|^./s)[0]; // an identifier as a whole (not a keyword in its middle)
				html += escape(word);
				text = text.slice(word.length);
			}
		}
		return html;
	}

	// Highlights a piece of text of a line; `state.inComment`: in a block comment that started on an earlier line.
	function highlightText(text, state) {
		let html = "";
		if (state.inComment) {
			const end = text.indexOf("*/");
			const comment = end < 0 ? text : text.slice(0, end + 2);
			html = `<span class="comment">${escape(comment)}</span>`;
			text = text.slice(comment.length);
			state.inComment = end < 0;
			if (state.inComment) return html;
		}
		const start = text.lastIndexOf("/*");
		if (start >= 0 && text.indexOf("*/", start) < 0) { // a block comment that continues on the next line
			state.inComment = true;
			return html + highlight(text.slice(0, start)) + `<span class="comment">${escape(text.slice(start))}</span>`;
		}
		return html + highlight(text);
	}

	// Highlights the lines of a code block. The text is colored around the elements of a line: its number (source
	// pages of mdl-doc) is kept, and the text of links (names linked to their documentation) is colored inside them.
	function highlightBlock(fragment) {
		const state = { inComment: false };
		for (const line of fragment.querySelectorAll(".line"))
			for (const node of [...line.childNodes]) {
				if (node.nodeType === Node.TEXT_NODE) {
					const span = document.createElement("span");
					span.innerHTML = highlightText(node.textContent, state);
					node.replaceWith(...span.childNodes);
				} else if (node.tagName === "A") {
					node.innerHTML = highlightText(node.textContent, state);
				}
			}
	}

	document.addEventListener("DOMContentLoaded", () => {
		for (const fragment of document.querySelectorAll(".mdl .fragment"))
			highlightBlock(fragment);
	});
})();
