# The documentation (target mdl_docs): a Doxygen site of
#  - the hand-written pages in docs/ (user docs, foundations, developer docs),
#  - the API of the standard library, generated from std/*.dlm by mdl-doc (src/commands/mdl-doc.cpp), and
#  - the C++ API in include/.
# The Doxygen configuration is set here (DOXYGEN_* variables) instead of a Doxyfile. The theme is doxygen-awesome-css.
# Citations (\cite key) come from all docs/references/*.bib files.
find_package(Doxygen 1.10 REQUIRED dot)

option(MDL_DOCS_WARN_AS_ERROR "Fail the documentation build on Doxygen warnings (CI)" OFF)

CPMAddPackage(
	NAME doxygen-awesome-css
	GITHUB_REPOSITORY jothepro/doxygen-awesome-css
	VERSION 2.5.0
	DOWNLOAD_ONLY YES
)
set(_awesome "${doxygen-awesome-css_SOURCE_DIR}")

# The tag file of cppreference.com: links from std:: names in the C++ API to cppreference. It is part of the offline
# archive of cppreference-doc (the file on cppreference.com itself is no longer downloadable); pinned by hash.
CPMAddPackage(
	NAME cppreference-tags
	URL https://github.com/PeterFeicht/cppreference-doc/releases/download/v20250209/html-book-20250209.zip
	URL_HASH SHA256=5389f2635f1417f05319b11f065ceecd1a8442c9143e2c127409137396b0c81c
	DOWNLOAD_ONLY YES
)

# ── The standard library's API pages (generated) ──────────────────────────────────────────────────────────────────
set(MDL_STD_API_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/std")
file(GLOB_RECURSE _std_sources CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/std/*.dlm")
add_custom_command(
	OUTPUT "${MDL_STD_API_DIR}/std.md"
	COMMAND ${CMAKE_COMMAND} -E rm -rf "${MDL_STD_API_DIR}"
	COMMAND mdl-doc "${PROJECT_SOURCE_DIR}/std" --output "${MDL_STD_API_DIR}" --title "Standard library API"
	DEPENDS mdl-doc ${_std_sources}
	COMMENT "Generating the API pages of the standard library"
	VERBATIM
)
add_custom_target(mdl_std_api DEPENDS "${MDL_STD_API_DIR}/std.md")

# ── Bibliography: every docs/references/*.bib (one key namespace; duplicates are an error) ─────────────────────────
file(GLOB _bib_files CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/docs/references/*.bib")
add_custom_target(
	mdl_check_bib
	COMMAND ${CMAKE_COMMAND} "-DBIB_FILES=${_bib_files}" -P "${PROJECT_SOURCE_DIR}/cmake/CheckBibKeys.cmake"
	COMMENT "Checking the bibliography for duplicate keys"
	VERBATIM
)

# ── Doxygen configuration ─────────────────────────────────────────────────────────────────────────────────────────
set(DOXYGEN_PROJECT_NAME "MDL")
set(DOXYGEN_PROJECT_NUMBER "${MDL_DOCS_VERSION}")
if(NOT DOXYGEN_PROJECT_NUMBER)
	set(DOXYGEN_PROJECT_NUMBER "${MDL_VERSION}")
endif()
set(DOXYGEN_PROJECT_BRIEF "Model Description Language")
set(DOXYGEN_PROJECT_LOGO "${PROJECT_SOURCE_DIR}/docs/media/icon-solo.png")
set(DOXYGEN_PROJECT_ICON "${PROJECT_SOURCE_DIR}/docs/media/icon-solo.png")
set(DOXYGEN_TAGFILES "${cppreference-tags_SOURCE_DIR}/cppreference-doxygen-web.tag.xml=https://en.cppreference.com/w/")
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation")
set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}/include")
set(DOXYGEN_STRIP_FROM_INC_PATH "${PROJECT_SOURCE_DIR}/include")
set(DOXYGEN_FULL_PATH_NAMES YES)
set(DOXYGEN_EXCLUDE_PATTERNS "*/generated/*" "*/docs/tutorials/*")
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
set(DOXYGEN_WARN_NO_PARAMDOC NO)
set(DOXYGEN_WARN_IF_INCOMPLETE_DOC NO)
if(MDL_DOCS_WARN_AS_ERROR)
	set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
endif()
set(DOXYGEN_MARKDOWN_ID_STYLE GITHUB)
set(DOXYGEN_TOC_INCLUDE_HEADINGS 4)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/docs/index.md")
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_VERSION MathJax_3)
set(DOXYGEN_MATHJAX_EXTENSIONS ams)
set(DOXYGEN_IMAGE_PATH "${PROJECT_SOURCE_DIR}/docs/media")
set(DOXYGEN_EXAMPLE_PATH "${PROJECT_SOURCE_DIR}/examples" "${PROJECT_SOURCE_DIR}/std")
set(DOXYGEN_CITE_BIB_FILES ${_bib_files})
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_HTML_CODE_FOLDING NO) # its "fold all" button would appear on the source pages of mdl-doc
# The todo list (and the \todo notes on the pages) only in development versions, not in the documentation of a release
if(MDL_DOCS_VERSION MATCHES "^v[0-9]+\\.[0-9]+") # a release (set by .github/workflows/docs.yml)
	set(DOXYGEN_GENERATE_TODOLIST NO)
else()
	set(DOXYGEN_GENERATE_TODOLIST YES)
endif()
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_QUIET YES)

# doxygen-awesome: sidebar-only layout with a dark-mode toggle (see its README)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
set(DOXYGEN_HTML_COLORSTYLE LIGHT)
set(DOXYGEN_HTML_HEADER "${PROJECT_SOURCE_DIR}/docs/doxygen/header.html")
set(DOXYGEN_LAYOUT_FILE "${PROJECT_SOURCE_DIR}/docs/doxygen/layout.xml")
set(DOXYGEN_HTML_EXTRA_STYLESHEET
	"${_awesome}/doxygen-awesome.css"
	"${_awesome}/doxygen-awesome-sidebar-only.css"
	"${_awesome}/doxygen-awesome-sidebar-only-darkmode-toggle.css"
	"${PROJECT_SOURCE_DIR}/docs/doxygen/mdl.css"
)
set(DOXYGEN_HTML_EXTRA_FILES
	"${_awesome}/doxygen-awesome-darkmode-toggle.js"
	"${_awesome}/doxygen-awesome-fragment-copy-button.js"
	"${_awesome}/doxygen-awesome-interactive-toc.js"
	"${_awesome}/doxygen-awesome-paragraph-link.js"
	"${PROJECT_SOURCE_DIR}/docs/doxygen/mdl-highlight.js"
	"${PROJECT_SOURCE_DIR}/docs/doxygen/mdl-version.js"
	"${PROJECT_SOURCE_DIR}/docs/doxygen/lang-highlight.js"
)

doxygen_add_docs(
	mdl_docs
	# in the order of the navigation: the main page, user guide, foundations, developer documentation
	"${PROJECT_SOURCE_DIR}/docs/index.md" "${PROJECT_SOURCE_DIR}/docs/user" "${PROJECT_SOURCE_DIR}/docs/foundations"
	"${PROJECT_SOURCE_DIR}/docs/dev" "${PROJECT_SOURCE_DIR}/include" "${MDL_STD_API_DIR}"
	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	COMMENT "Generating the documentation"
)
add_dependencies(mdl_docs mdl_std_api mdl_check_bib)
