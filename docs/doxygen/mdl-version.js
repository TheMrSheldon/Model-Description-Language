/**
 * Turns the version number in the sidebar into a dropdown of the published versions.
 *
 * The published site has one directory per version (vX.Y.Z/, latest/, nightly/) and a versions.json next to them
 * that lists the directories. The documentation root of this page is one of these directories, so versions.json is
 * at "<relpath>../versions.json". Without it (e.g. a local build) the plain version number stays.
 */
function mdlVersionSelector(relpath) {
	const root = new URL(relpath || "./", document.baseURI);
	const current = root.pathname.split("/").filter(Boolean).pop();
	fetch(new URL("../versions.json", root))
		.then(response => (response.ok ? response.json() : Promise.reject()))
		.then(versions => {
			if (!Array.isArray(versions) || !versions.includes(current)) return;
			const install = () => {
				const number = document.getElementById("projectnumber");
				if (!number) return;
				const select = document.createElement("select");
				select.id = "versionselect";
				select.setAttribute("aria-label", "Documentation version");
				for (const version of versions) select.add(new Option(version, version, false, version === current));
				select.addEventListener("change", () => {
					// The same page in the other version; versions without it show their 404 page.
					const page = location.href.slice(root.href.length);
					location.href = new URL("../" + select.value + "/" + page, root).href;
				});
				number.replaceChildren(select);
			};
			if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install);
			else install();
		})
		.catch(() => {});
}
