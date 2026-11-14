# Editor tooling: `mdl-format` and `mdls`

Besides the compiler (`mdlc`), the project builds two tools for editing `.dlm` files:

- **`mdl-format`** formats source files, like clang-format.
- **`mdls`** is a language server (Language Server Protocol) for editors.

Both use the same frontend as the compiler (library `mdl_frontend`: parser, type checks, diagnostics). They need
neither TVM/LLVM nor oneDNN, and build with each platform's native toolchain (Linux, macOS, Windows/MSVC; x86_64 and
arm64).

## Prebuilt binaries
Each release (and the rolling `nightly` release of `main`) has archives `mdl-tools-<platform>` for `linux-x64`,
`linux-arm64`, `darwin-universal` (arm64 and x86_64), `win32-x64`, and `win32-arm64`, plus `SHA256SUMS`. They contain
`bin/mdl-format`, `bin/mdls`, and `share/mdl/std/`. They have no dependencies besides the operating system:
- **Linux:** built on glibc 2.28 with libstdc++ and libgcc linked statically, so they run on distributions with glibc
  2.28 or newer (e.g. Debian 10, Ubuntu 18.10, RHEL 8).
- **Windows:** the static C runtime.
- **macOS:** macOS 13.3 or newer.

The version (`--version`, and `serverInfo` of `mdls`) comes from the latest git tag (`git describe --tags`; `0.0.0-dev`
without tags).

## Building only the tooling
```bash
cmake --preset conf-tooling -B build-tooling      # MDL_ENABLE_COMPILER=OFF
cmake --build build-tooling --parallel
cmake --install build-tooling --prefix ~/.local --component tools   # bin/mdl-format, bin/mdls, share/mdl/std
```
Code generation of the parser needs Java at build time. An installed tool finds the standard library in
`<prefix>/share/mdl/std`. Otherwise it looks in `MDL_STDLIB_PATH` (separated by `:`, `;` on Windows) and in the
source tree's `std/`.

## `mdl-format`
```bash
mdl-format model.dlm              # prints the formatted file
mdl-format -i *.dlm               # formats in place
mdl-format --check $(git ls-files '*.dlm')   # exit code 1 if a file is not formatted (CI)
cat model.dlm | mdl-format --assume-filename model.dlm
```
It changes only whitespace:
- **Indentation:** blocks are indented.
- **Line breaks:**
  - one definition, member or statement per line;
  - lines that are too long break after `=>`, then after the commas of the outermost list, then before binary
    operators.
- **Spaces:** around binary operators, `=>`, `->` and `|`; none in `.bias=false` or `h=heads`. Einops patterns are
  written as `x[... (h d) -> h ... d | h=heads]`.
- **Comments:** kept; block comments are re-indented as a whole, and consecutive trailing comments are aligned.
- **Kept line breaks:** after `template<...>` and after an annotation, if the source has them.
- **Blank lines:** at most `MaxEmptyLines` between definitions and statements; none at the start or end of a block.

A file with syntax errors is left unchanged (the errors are reported). The formatter checks its own output: it must
contain exactly the input's tokens.

The style comes from `--style-file`, else from the closest `.mdl-format` file in the source's directory or one of its
parents, else the defaults:

```yaml
# .mdl-format
IndentWidth: 4          # spaces per level; continuation lines get twice as many
ColumnLimit: 120
MaxEmptyLines: 1
ReflowComments: false   # true: comment lines longer than ColumnLimit are broken between words

# Naming conventions of definitions: snake_case, camelCase, PascalCase, ALL_CAPS, lowercase, or any (the default)
ModuleNaming: PascalCase
StructNaming: PascalCase
StructMemberNaming: camelCase
ModuleMemberNaming: camelCase       # weights and submodules
FunctionNaming: camelCase           # free and member functions
ParameterNaming: camelCase
HyperparameterNaming: camelCase
TemplateParameterNaming: any
VariableNaming: camelCase           # local and loop variables
```
Unknown keys and values are errors.

The naming conventions are checks, not changes: renaming a definition would need changes in every file that uses it.
A definition that breaks a rule is reported as a warning with a suggested name, e.g.
`warning: The hyperparameter name hiddenSize is not snake_case [-Wnaming]` with `hint: rename it to hidden_size`. Only
definitions are checked, not uses (e.g. of the standard library). `--check` fails on naming warnings, and `mdls` shows
them as diagnostics with the code `naming`.

## `mdls`
The editor starts `mdls` as a child process and talks to it over stdin/stdout; it exits with the editor. Options:
- `--log FILE`: write a log.
- `--stdio`: accepted for clients that pass it.

Features:
- **Diagnostics:** the compiler's errors and warnings while you type. The hint goes in the message, and the `-W` flag
  in the diagnostic's code.
- **Generic modules:** modules with hyperparameters are checked too, with symbolic values, so types read
  `float[..., dim]`.
  - What depends on unknown values is checked conservatively: both branches of `?:`, members with `@conditional`, one
    iteration of a loop, and one instance of an array of modules.
  - Only definite mismatches are errors. Sizes that are different hyperparameters (`indim` vs `outdim`) could be
    equal in an instance: they are warnings (`possible-mismatch`), and errors with the `strict` option.
- **Semantic tokens:** what needs the parse tree: modules, structs, functions and methods, hyperparameters, members,
  parameters, variables, annotations, and the axes of einops patterns (their own token type `axis`, which an editor
  extension declares with a fallback). Keywords, literals, and operators are left to the editor's grammar: they need no
  context, and a grammar can distinguish them further (e.g. control-flow keywords from declarations). In doc comments,
  with the modifier `documentation`: the Doxygen commands (`keyword`), parameter names after `@param`/`@tparam`/`\p`
  (`parameter`), the targets of `\ref`, `\see`, `\sa` and the keys of `\cite` (`label`), and math (`string`).
- **Hover:**
  - the declaration as written;
  - its kind and container;
  - the inferred type(s), e.g. for `auto` variables, symbolic and concrete (`float32[..., d]`,
    `float32[..., 768]`);
  - the documentation from Doxygen-style comments as Markdown. Commands are spelled with `@` or `\`:
    - paragraphs: `brief`, `details`, `param`/`tparam`, `return`, and the sections `note`, `warning`, `see`/`sa`,
      `deprecated`, `todo`;
    - inline: math (`f$`, `f(`, `f[`, `f{env}{`), `p` and `c` (code), `a`/`e`/`em` (italic), `b` (bold), `cite`;
    - code blocks: `code` ... `endcode`.

    Hyperparameters, template parameters, and function parameters show their `@param`/`@tparam` text from the comment
    of their module or function;
  - on an import path (`import std.nn;`): the file's `@file` comment, or for a package (`std`) its files with the
    first paragraph of their documentation.
- **Go to definition:** also into imported files and the standard library; on an import path, the imported file (or
  the files of a package).
- **Completion:**
  - names in scope;
  - members after `.` (module members, struct members, `data`/`mask` of masked tensors);
  - named hyperparameters after `.` in an instantiation (`Linear[4, 3, .bias=...]`);
  - `import std.` modules, keywords, types, annotations.

  While the text does not parse (e.g. right after typing `.`), the last analysis is used.
- **Hover on `auto`:** the inferred type of the declared name(s); on an einops axis its size; on a weight its full
  path (e.g. `layers.3.weightQuery.W`, as in checkpoints).
- **Signature help:** while typing the arguments of a call (`f(`, `inst.f(`, `inst(`) or the hyperparameters of an
  instantiation (`Linear[`): the declaration, the active parameter (by position, or by name for `.name=`), and the
  `@param` documentation.
- **Inlay hints:** the inferred type after `auto` declarations, the size of each einops axis (`(h=12 d=64)`), and the
  names of positional hyperparameters (`Linear[indim: 768, outdim: 3]`).
- **Find references, highlight, rename:** of variables, parameters, members, functions, modules, structs (across the
  open documents and their imports; definitions of the standard library are not renamed). Highlighting an einops axis
  marks its occurrences in the pattern.
- **Quick fixes:** `.name=value` for an optional hyperparameter given by position, `float32(x)` (or the respective
  type) for an implicit float conversion, renaming a definition that breaks a naming rule, and an `import std.x;` for
  an undefined name that the standard library defines.
- **Go to definition of struct members** (`conf.dims.model`), **folding ranges** (blocks, comments, imports), and
  **range formatting**.
- **Formatting:** the whole document, with `mdl-format`'s style (from `.mdl-format`, else the editor's tab size).
- **Document symbols:** the outline of modules, structs and functions with their members.

Documents are synchronized as full text and re-analyzed on every change (about 15 ms for the BERT example), together
with the open documents that import them. Open documents take precedence over the files on disk, also for imports.
While a document does not parse, hover, go to definition, and completion use its last analysis (positions after the
changed lines are mapped).

Initialization options (from the editor), the same as `mdlc`'s:
```json
{ "importDirs": ["path/to/lib"], "float": "bf16", "strict": true }
```
With `strict`, sizes that may differ (e.g. two hyperparameters where equal sizes are needed) are errors instead of
warnings.

### Using it from an editor without an extension
Any LSP client can run it, e.g. Neovim:
```lua
vim.filetype.add({ extension = { dlm = "mdl" } })
vim.lsp.start({ name = "mdls", cmd = { "mdls" }, root_dir = vim.fs.root(0, { ".git" }) })
```
A VS Code extension (a TextMate grammar for basic highlighting plus this server) is planned in a separate repository.

## Library structure
| Target | Contents | Dependencies |
|---|---|---|
| `mdl_frontend` | parser, diagnostics (`mdl::Diagnostic`), type checks, symbolic analysis, semantic index | ANTLR runtime |
| `mdl_format` | the formatter (`mdl::format::format`) | `mdl_frontend` |
| `mdls_lib` | the protocol, document handling and features of `mdls` | `mdl_frontend`, `mdl_format`, nlohmann/json |
| `mdl` | passes (masking, autodiff, quantization), exports, bindings, TVM backend | `mdl_frontend`, TVM, yaml-cpp, spdlog |

Tools use `mdl::parser::analyze()`, which keeps the compilation unit and an `analysis::SemanticIndex` (definitions,
references, types, documentation) also when there are errors. The compiler uses `parse()`, which prints the
diagnostics (or passes them to `ParseOptions::diagnostics`).
