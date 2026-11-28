// Reading and validating package manifests (package.yml, schema 1; see docs/dev/spec/kernel-packages.md).
#include <mdl/kernels.hpp>
#include <mdl/kernels/expr.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <format>
#include <functional>
#include <set>
#include <stdexcept>

namespace mdl::kernels {

	std::string Problem::str() const {
		if (line == 0)
			return std::format("{}: {}", file.string(), message);
		return std::format("{}:{}:{}: {}", file.string(), line, column, message);
	}

	std::string Platform::str() const {
		std::string s = os + " " + arch;
		if (!abi.empty())
			s += " " + abi;
		if (!libc.empty())
			s += " libc" + libc;
		return s;
	}

	const Artifact* Package::artifact(const std::string& id) const {
		auto it = std::ranges::find(artifacts, id, &Artifact::id);
		return it == artifacts.end() ? nullptr : &*it;
	}

	const Pattern* Package::pattern(const std::string& id) const {
		auto it = std::ranges::find(patterns, id, &Pattern::id);
		return it == patterns.end() ? nullptr : &*it;
	}

	namespace {
		/** @brief Thrown for a problem that makes the rest of the manifest meaningless (e.g. a wrong schema version). */
		struct Fatal {};

		/** @brief Reads one manifest and collects its problems. */
		class Reader {
		public:
			Reader(std::filesystem::path file, std::vector<Problem>& problems)
					: file(std::move(file)), problems(problems) {}

			size_t errors = 0;

			void problem(const YAML::Node& at, std::string message) {
				const auto mark = at.Mark();
				const bool known = mark.line >= 0 && !at.IsNull();
				problems.push_back(
						Problem{.file = file,
								.line = known ? static_cast<size_t>(mark.line) + 1 : 0,
								.column = known ? static_cast<size_t>(mark.column) + 1 : 0,
								.message = std::move(message)}
				);
				++errors;
			}

			static size_t lineOf(const YAML::Node& n) { return n.Mark().line >= 0 ? n.Mark().line + 1 : 0; }

			/** @brief Reports the keys of @p map that are not in @p allowed. */
			void
			onlyKeys(const YAML::Node& map, std::initializer_list<std::string_view> allowed, std::string_view where) {
				for (const auto& kv : map) {
					const auto key = kv.first.as<std::string>();
					if (std::ranges::find(allowed, key) == allowed.end()) {
						std::string list;
						for (auto a : allowed)
							list += (list.empty() ? "" : ", ") + std::string(a);
						problem(kv.first, std::format("unknown key '{}' in {} (expected: {})", key, where, list));
					}
				}
			}

			bool isMap(const YAML::Node& n, std::string_view what) {
				if (n.IsMap())
					return true;
				problem(n, std::format("{} must be a map", what));
				return false;
			}

			std::string string(const YAML::Node& n, std::string_view what) {
				if (n.IsScalar())
					return n.as<std::string>();
				problem(n, std::format("{} must be a string", what));
				return {};
			}

			std::string requiredString(const YAML::Node& map, const char* key, std::string_view where) {
				const auto n = map[key];
				if (!n) {
					problem(map, std::format("{} needs '{}'", where, key));
					return {};
				}
				return string(n, std::format("'{}' of {}", key, where));
			}

			std::optional<int64_t> integer(const YAML::Node& n, std::string_view what) {
				int64_t v = 0;
				if (n.IsScalar() && YAML::convert<int64_t>::decode(n, v))
					return v;
				problem(n, std::format("{} must be an integer", what));
				return std::nullopt;
			}

			std::optional<double> number(const YAML::Node& n, std::string_view what) {
				double v = 0;
				if (n.IsScalar() && YAML::convert<double>::decode(n, v))
					return v;
				problem(n, std::format("{} must be a number", what));
				return std::nullopt;
			}

			std::vector<std::string> strings(const YAML::Node& n, std::string_view what) {
				std::vector<std::string> result;
				if (n.IsScalar()) {
					result.push_back(n.as<std::string>());
				} else if (n.IsSequence()) {
					for (const auto& item : n)
						result.push_back(string(item, what));
				} else {
					problem(n, std::format("{} must be a string or a list of strings", what));
				}
				return result;
			}

			/** @brief One of @p choices (an error otherwise). */
			std::string
			choice(const YAML::Node& n, std::string_view what, std::initializer_list<std::string_view> choices) {
				const auto value = string(n, what);
				if (value.empty() || std::ranges::find(choices, value) != choices.end())
					return value;
				std::string list;
				for (auto c : choices)
					list += (list.empty() ? "" : ", ") + std::string(c);
				problem(n, std::format("{} '{}' is not one of: {}", what, value, list));
				return {};
			}

			/** @brief An id of a section entry: letters, digits, `-`, `_`, `.`. */
			bool validId(const YAML::Node& key, const std::string& id) {
				const bool ok = !id.empty() && std::ranges::all_of(id, [](char c) {
					return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.';
				});
				if (!ok)
					problem(key, std::format("invalid id '{}' (letters, digits, '-', '_', '.')", id));
				return ok;
			}

			Platform platform(const YAML::Node& n) {
				Platform p;
				if (!isMap(n, "'platform'"))
					return p;
				onlyKeys(n, {"os", "arch", "abi", "libc"}, "platform");
				p.os = n["os"] ? choice(n["os"], "os", {"linux", "windows", "darwin"}) : "";
				p.arch = n["arch"] ? choice(n["arch"], "arch", {"x86_64", "aarch64"}) : "";
				p.abi = n["abi"] ? choice(n["abi"], "abi", {"gnu", "msvc", "darwin"}) : "";
				p.libc = n["libc"] ? string(n["libc"], "libc") : "";
				if (p.os.empty() || p.arch.empty())
					problem(n, "a platform needs 'os' and 'arch'");
				if (!p.libc.empty() && !p.libc.starts_with(">="))
					problem(n["libc"], "libc must be a minimum version like '>=2.28'");
				return p;
			}

			Artifact artifact(const std::string& id, const YAML::Node& n, const std::filesystem::path& dir) {
				Artifact a{.id = id, .kind = Artifact::Kind::staticLibrary, .line = lineOf(n)};
				if (!isMap(n, std::format("artifact '{}'", id)))
					return a;
				onlyKeys(n, {"kind", "path", "platform", "cpu-features", "device-arch", "dependencies"}, "an artifact");
				const auto kind = choice(n["kind"], "kind", {"static-library", "shared-library", "cubin", "ptx"});
				if (!n["kind"])
					problem(n, std::format("artifact '{}' needs 'kind'", id));
				a.kind = kind == "shared-library" ? Artifact::Kind::sharedLibrary
						 : kind == "cubin"		  ? Artifact::Kind::cubin
						 : kind == "ptx"		  ? Artifact::Kind::ptx
												  : Artifact::Kind::staticLibrary;
				const auto path = requiredString(n, "path", std::format("artifact '{}'", id));
				a.path = path.empty() ? std::filesystem::path{} : (dir / path).lexically_normal();
				if (n["platform"])
					a.platform = platform(n["platform"]);
				else if (a.supported() && !kind.empty())
					problem(n, std::format("artifact '{}' needs 'platform' (binary compatibility)", id));
				if (n["cpu-features"])
					a.cpuFeatures = strings(n["cpu-features"], "cpu-features");
				if (n["device-arch"])
					a.deviceArch = string(n["device-arch"], "device-arch");
				if (const auto deps = n["dependencies"]) {
					if (!deps.IsSequence())
						problem(deps, "'dependencies' must be a list");
					for (const auto& d : deps) {
						if (!isMap(d, "a dependency"))
							continue;
						onlyKeys(d, {"kind", "name"}, "a dependency");
						const auto k = choice(d["kind"], "kind", {"system-library", "runtime", "artifact"});
						const auto name = requiredString(d, "name", "a dependency");
						if (k == "runtime" && name != "openmp")
							problem(d["name"], std::format("unknown runtime '{}' (known: openmp)", name));
						a.dependencies.push_back(Dependency{
								.kind = k == "runtime"	  ? Dependency::Kind::runtime
										: k == "artifact" ? Dependency::Kind::artifact
														  : Dependency::Kind::systemLibrary,
								.name = name
						});
					}
				}
				return a;
			}

			PatternNode patternNode(const YAML::Node& n, std::set<std::string>& captures) {
				PatternNode node{.line = lineOf(n)};
				if (!isMap(n, "a pattern node"))
					return node;
				if (n["capture"] && !n["op"]) { // (an operator node may name its result with `capture:` too)
					onlyKeys(n, {"capture", "not"}, "a capture");
					node.kind = PatternNode::Kind::capture;
					node.capture = string(n["capture"], "capture");
					if (const auto no = n["not"]) { // e.g. `not: relax.multiply` (fused with it by TVM instead)
						if (no.IsSequence())
							for (const auto& op : no)
								node.notOps.push_back(string(op, "not"));
						else
							node.notOps.push_back(string(no, "not"));
					}
					captures.insert(node.capture);
				} else if (n["const"]) {
					onlyKeys(n, {"const", "tolerance"}, "a constant");
					node.kind = PatternNode::Kind::constant;
					// A number, or an integer expression over the captures (e.g. the size that a mean divides by).
					double v = 0;
					if (n["const"].IsScalar() && YAML::convert<double>::decode(n["const"], v))
						node.value = v;
					else
						node.valueExpr = string(n["const"], "const");
					if (n["tolerance"])
						node.tolerance = number(n["tolerance"], "tolerance");
				} else if (n["op"]) {
					onlyKeys(n, {"op", "args", "attrs", "capture"}, "an operator node");
					node.kind = PatternNode::Kind::op;
					node.op = string(n["op"], "op");
					if (n["capture"]) { // the operator's result (and its attributes: $name.attr.<name>)
						node.capture = string(n["capture"], "capture");
						captures.insert(node.capture);
					}
					if (const auto args = n["args"]) {
						if (!args.IsSequence())
							problem(args, "'args' must be a list");
						for (const auto& arg : args)
							node.args.push_back(patternNode(arg, captures));
					}
					if (const auto attrs = n["attrs"]) {
						if (isMap(attrs, "'attrs'"))
							for (const auto& kv : attrs) {
								// Lists (e.g. axes) are kept in flow style: `[-1]`.
								YAML::Emitter out;
								out << YAML::Flow << kv.second;
								node.attrs[kv.first.as<std::string>()] = out.c_str();
							}
					}
				} else {
					problem(n, "a pattern node needs 'op', 'capture', or 'const'");
				}
				return node;
			}

			Pattern pattern(const std::string& id, const YAML::Node& n) {
				Pattern p{.id = id, .line = lineOf(n)};
				if (!isMap(n, std::format("pattern '{}'", id)))
					return p;
				std::set<std::string> captures;
				// The root is the pattern itself (an operator node), with optional `outputs`.
				YAML::Node root = YAML::Clone(n);
				if (root["outputs"]) {
					p.outputs = strings(root["outputs"], "outputs");
					root.remove("outputs");
				}
				p.root = patternNode(root, captures);
				if (p.root.kind != PatternNode::Kind::op)
					problem(n, std::format("pattern '{}' must be an operator node (with 'op')", id));
				for (const auto& out : p.outputs)
					if (!captures.contains(out))
						problem(n["outputs"], std::format("output '{}' of pattern '{}' is not a capture", out, id));
				return p;
			}

			Implementation implementation(const std::string& id, const YAML::Node& n) {
				Implementation impl{.id = id, .line = lineOf(n)};
				const std::string where = std::format("implementation '{}'", id);
				if (!isMap(n, where))
					return impl;
				onlyKeys(
						n,
						{"pattern", "artifacts", "priority", "symbol", "abi", "signature", "applicability", "guarantees"
						},
						"an implementation"
				);
				impl.pattern = requiredString(n, "pattern", where);
				if (n["artifacts"])
					impl.artifacts = strings(n["artifacts"], "artifacts");
				else
					problem(n, where + " needs 'artifacts'");
				if (n["priority"])
					impl.priority = integer(n["priority"], "priority").value_or(0);
				impl.symbol = requiredString(n, "symbol", where);
				if (n["abi"])
					impl.abi = choice(n["abi"], "abi", {"c"});
				if (const auto sig = n["signature"]; sig && isMap(sig, "'signature'")) {
					onlyKeys(sig, {"returns", "arguments", "aliasing", "outputs"}, "a signature");
					if (const auto ret = sig["returns"]) {
						if (isMap(ret, "'returns'")) {
							onlyKeys(ret, {"type", "error"}, "'returns'");
							impl.returns = choice(ret["type"], "type", {"int32"});
							if (ret["error"])
								choice(ret["error"], "error", {"nonzero"});
						}
					}
					if (const auto args = sig["arguments"]) {
						if (!args.IsSequence())
							problem(args, "'arguments' must be a list");
						for (const auto& a : args) {
							if (!isMap(a, "an argument"))
								continue;
							onlyKeys(a, {"type", "access", "value", "align"}, "an argument");
							Argument arg;
							arg.type =
									choice(a["type"], "type",
										   {"pointer", "int64", "int32", "float32", "float64", "int64[]"});
							arg.value = requiredString(a, "value", "an argument");
							if (arg.type == "pointer") {
								if (a["access"])
									arg.access = choice(a["access"], "access", {"read", "write", "readwrite"});
								else
									problem(a, "a pointer argument needs 'access' (read, write, readwrite)");
							} else if (a["access"]) {
								problem(a["access"], "'access' applies to pointer arguments only");
							}
							if (a["align"])
								arg.align = integer(a["align"], "align").value_or(0);
							impl.arguments.push_back(std::move(arg));
						}
					}
					if (const auto al = sig["aliasing"]) {
						if (al.IsScalar())
							impl.aliasing = choice(al, "aliasing", {"none"});
						else if (al.IsMap() && al["out"])
							impl.aliasing = "out=" + string(al["out"], "out");
						else
							problem(al, "'aliasing' must be 'none' or {out: <capture>}");
					}
					if (const auto outs = sig["outputs"]) {
						if (!outs.IsSequence())
							problem(outs, "'outputs' must be a list");
						for (const auto& o : outs) {
							if (!isMap(o, "an output"))
								continue;
							onlyKeys(o, {"shape", "dtype"}, "an output");
							impl.outputs.emplace_back(
									requiredString(o, "shape", "an output"), requiredString(o, "dtype", "an output")
							);
						}
					}
				}
				if (impl.arguments.empty())
					problem(n, where + " needs 'signature.arguments'");
				if (const auto app = n["applicability"]; app && isMap(app, "'applicability'")) {
					onlyKeys(app, {"dtype", "layout", "where"}, "'applicability'");
					if (const auto d = app["dtype"]; d && isMap(d, "'dtype'"))
						for (const auto& kv : d)
							impl.dtypes[kv.first.as<std::string>()] = strings(kv.second, "a dtype");
					if (const auto l = app["layout"]; l && isMap(l, "'layout'"))
						for (const auto& kv : l)
							impl.layouts[kv.first.as<std::string>()] = choice(kv.second, "layout", {"row-major"});
					if (app["where"])
						impl.where = strings(app["where"], "where");
				}
				if (const auto g = n["guarantees"]; g && isMap(g, "'guarantees'"))
					flatten(g, "", impl.guarantees);
				return impl;
			}

			/** @brief Nested maps as dotted keys (`numerics.max_ulp: 2`). */
			void flatten(const YAML::Node& n, const std::string& prefix, std::map<std::string, std::string>& out) {
				for (const auto& kv : n) {
					const auto key = prefix + kv.first.as<std::string>();
					if (kv.second.IsMap())
						flatten(kv.second, key + ".", out);
					else if (kv.second.IsScalar())
						out[key] = kv.second.as<std::string>();
					else
						problem(kv.second, std::format("'{}' must be a value or a map", key));
				}
			}

			std::optional<Package> package(const YAML::Node& root) {
				if (!root.IsMap()) {
					problem(root, "the manifest must be a map");
					return std::nullopt;
				}
				onlyKeys(
						root,
						{"schema", "package", "version", "description", "artifacts", "patterns", "implementations"},
						"the manifest"
				);
				const auto schema = root["schema"];
				if (!schema) {
					problem(root, std::format("the manifest needs 'schema: {}'", kSchemaVersion));
					throw Fatal{};
				}
				if (auto v = integer(schema, "schema"); !v || *v != kSchemaVersion) {
					if (v)
						problem(schema, std::format(
												"unsupported schema version {} (this compiler reads schema {})", *v,
												kSchemaVersion
										));
					throw Fatal{};
				}
				Package p;
				p.manifest = file;
				p.directory = file.parent_path();
				p.name = requiredString(root, "package", "the manifest");
				if (!p.name.empty())
					validId(root["package"], p.name);
				p.version = requiredString(root, "version", "the manifest");
				if (root["description"])
					p.description = string(root["description"], "description");

				auto section = [&](const char* name, auto&& read) {
					const auto n = root[name];
					if (!n)
						return;
					if (!isMap(n, std::format("'{}'", name)))
						return;
					for (const auto& kv : n) {
						const auto id = kv.first.as<std::string>();
						if (validId(kv.first, id))
							read(id, kv.second);
					}
				};
				section("artifacts", [&](const std::string& id, const YAML::Node& n) {
					p.artifacts.push_back(artifact(id, n, p.directory));
				});
				section("patterns",
						[&](const std::string& id, const YAML::Node& n) { p.patterns.push_back(pattern(id, n)); });
				section("implementations", [&](const std::string& id, const YAML::Node& n) {
					p.implementations.push_back(implementation(id, n));
				});

				// References between the sections.
				for (const auto& impl : p.implementations) {
					const auto at = root["implementations"][impl.id];
					if (!impl.pattern.empty() && p.pattern(impl.pattern) == nullptr)
						problem(at["pattern"],
								std::format(
										"implementation '{}' refers to an unknown pattern '{}'", impl.id, impl.pattern
								));
					for (const auto& a : impl.artifacts)
						if (p.artifact(a) == nullptr)
							problem(at["artifacts"],
									std::format("implementation '{}' refers to an unknown artifact '{}'", impl.id, a));
				}
				for (const auto& impl : p.implementations)
					if (const auto* pattern = p.pattern(impl.pattern))
						checkExpressions(impl, *pattern, root["implementations"][impl.id]);
				for (const auto& a : p.artifacts)
					for (const auto& d : a.dependencies)
						if (d.kind == Dependency::Kind::artifact && p.artifact(d.name) == nullptr)
							problem(root["artifacts"][a.id]["dependencies"],
									std::format("artifact '{}' depends on an unknown artifact '{}'", a.id, d.name));
				return p;
			}

			static void collectCaptures(const PatternNode& node, std::vector<std::string>& out) {
				if (!node.capture.empty() && std::ranges::find(out, node.capture) == out.end())
					out.push_back(node.capture);
				for (const auto& arg : node.args)
					collectCaptures(arg, out);
			}

			/** @brief Parses the expressions of @p impl over the captures of @p pattern and checks their types. */
			void checkExpressions(const Implementation& impl, const Pattern& pattern, const YAML::Node& at) {
				std::vector<std::string> captures;
				collectCaptures(pattern.root, captures);
				auto check = [&](const std::string& text, expr::Type type, const YAML::Node& node,
								 std::string_view what) {
					auto parsed = expr::parse(text, captures);
					if (!parsed)
						problem(node, std::format("{} '{}': {}", what, text, parsed.error()));
					else if ((*parsed)->type != type)
						problem(node, std::format(
											  "{} '{}' must be {}, not {}", what, text, expr::str(type),
											  expr::str((*parsed)->type)
									  ));
				};
				const auto sig = at["signature"];
				for (size_t i = 0; i < impl.arguments.size(); ++i) {
					const auto& arg = impl.arguments[i];
					const bool real = arg.type == "float32" || arg.type == "float64";
					auto parsed = real ? expr::parse(arg.value, captures) : std::expected<expr::NodePtr, std::string>{};
					if (real && parsed && (*parsed)->type == expr::Type::integer)
						continue; // an integer converts to a float argument
					check(arg.value,
						  arg.type == "pointer"	  ? expr::Type::pointer
						  : arg.type == "int64[]" ? expr::Type::shape
						  : real				  ? expr::Type::real
												  : expr::Type::integer,
						  sig["arguments"][i]["value"], std::format("argument {}", i + 1));
				}
				for (size_t i = 0; i < impl.outputs.size(); ++i) {
					check(impl.outputs[i].first, expr::Type::shape, sig["outputs"][i]["shape"], "the output shape");
					check(impl.outputs[i].second, expr::Type::dtype, sig["outputs"][i]["dtype"], "the output dtype");
				}
				// Constants given as expressions.
				std::function<void(const PatternNode&)> constants = [&](const PatternNode& node) {
					if (!node.valueExpr.empty()) {
						auto parsed = expr::parse(node.valueExpr, captures);
						if (!parsed)
							problems.push_back(
									Problem{.file = file,
											.line = node.line,
											.column = 0,
											.message = std::format("constant '{}': {}", node.valueExpr, parsed.error())}
							);
						else if ((*parsed)->type != expr::Type::integer)
							problems.push_back(
									Problem{.file = file,
											.line = node.line,
											.column = 0,
											.message = std::format(
													"constant '{}' must be an integer, not {}", node.valueExpr,
													expr::str((*parsed)->type)
											)}
							);
						errors += parsed && (*parsed)->type == expr::Type::integer ? 0 : 1;
					}
					for (const auto& arg : node.args)
						constants(arg);
				};
				constants(pattern.root);
				const auto app = at["applicability"];
				const auto where = app ? app["where"] : YAML::Node{};
				for (size_t i = 0; i < impl.where.size(); ++i)
					check(impl.where[i], expr::Type::boolean, where.IsSequence() ? where[i] : where, "the condition");
				auto known = [&](const std::map<std::string, auto>& entries, const char* section) {
					for (const auto& [name, _] : entries)
						if (std::ranges::find(captures, name) == captures.end())
							problem(at["applicability"][section],
									std::format(
											"'{}' in {} is not a capture of pattern '{}'", name, section, pattern.id
									));
				};
				known(impl.dtypes, "dtype");
				known(impl.layouts, "layout");
				if (impl.aliasing.starts_with("out=") &&
					std::ranges::find(captures, impl.aliasing.substr(4)) == captures.end())
					problem(sig["aliasing"],
							std::format("aliasing refers to an unknown capture '{}'", impl.aliasing.substr(4)));
			}

		private:
			std::filesystem::path file;
			std::vector<Problem>& problems;
		};
	} // namespace

	namespace {
		/**
		 * @brief @p node with YAML merge keys expanded (`<<: *anchor` copies the anchor's keys that the map does not
		 * have; yaml-cpp does not implement them), e.g. for implementations that differ only in the symbol.
		 */
		YAML::Node expandMerges(YAML::Node node) {
			// In place (the loaded document is ours): the nodes keep their positions for the problems.
			if (node.IsSequence()) {
				for (auto item : node)
					expandMerges(item);
				return node;
			}
			if (!node.IsMap())
				return node;
			if (const auto merge = node["<<"]) {
				std::vector<YAML::Node> sources;
				if (merge.IsSequence())
					for (const auto& m : merge)
						sources.push_back(m);
				else
					sources.push_back(merge);
				node.remove("<<");
				for (auto& source : sources)
					if (source.IsMap())
						for (const auto& kv : expandMerges(source))
							if (!node[kv.first.as<std::string>()])
								node[kv.first.as<std::string>()] = kv.second;
			}
			for (auto kv : node)
				expandMerges(kv.second);
			return node;
		}
	} // namespace

	std::optional<Package> loadPackage(const std::filesystem::path& file, std::vector<Problem>& problems) {
		const size_t before = problems.size();
		Reader reader{file, problems};
		// The problems of this manifest in the order of their positions.
		auto sorted = [&] {
			std::stable_sort(
					problems.begin() + static_cast<std::ptrdiff_t>(before), problems.end(),
					[](const auto& a, const auto& b) {
						return std::pair{a.line, a.column} < std::pair{b.line, b.column};
					}
			);
		};
		try {
			const YAML::Node root = expandMerges(YAML::LoadFile(file.string()));
			auto package = reader.package(root);
			sorted();
			if (reader.errors > 0)
				return std::nullopt;
			return package;
		} catch (const YAML::Exception& e) {
			problems.push_back(
					Problem{.file = file,
							.line = e.mark.line >= 0 ? static_cast<size_t>(e.mark.line) + 1 : 0,
							.column = e.mark.column >= 0 ? static_cast<size_t>(e.mark.column) + 1 : 0,
							.message = e.msg}
			);
		} catch (const Fatal&) {
		}
		return std::nullopt;
	}

} // namespace mdl::kernels
