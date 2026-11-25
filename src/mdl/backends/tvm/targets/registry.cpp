#include "./registry.hpp"

#include "./cuda.hpp"
#include "./llvm.hpp"

#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mdl::backend::tvm {

	namespace {
		using Factory = std::function<std::unique_ptr<Target>(const mdl::CompileOptions&)>;

		template <typename T>
		Factory factory() {
			return [](const mdl::CompileOptions& opts) { return std::make_unique<T>(opts); };
		}

		const std::map<std::string, Factory>& registry() {
			static const std::map<std::string, Factory> targets{
					{"llvm", factory<LLVMTarget>()},
					{"cuda", factory<CUDATarget>()},
			};
			return targets;
		}

		std::string jsonString(const std::string& s) {
			std::string out = "\"";
			for (char c : s)
				out += c == '"' || c == '\\' ? std::string{'\\', c} : std::string{c};
			return out + "\"";
		}

	} // namespace

	::tvm::Target parseTarget(const std::string& spec) {
		if (spec.empty() || spec.front() == '{' || spec.find(' ') == std::string::npos)
			return ::tvm::Target(spec);
		std::istringstream words(spec);
		std::string kind, option;
		words >> kind;
		std::string json = "{\"kind\": " + jsonString(kind);
		while (words >> option) {
			const auto eq = option.find('=');
			if (!option.starts_with('-') || eq == std::string::npos)
				throw std::runtime_error("target '" + spec + "': expected options -key=value, got " + option);
			const std::string key = option.substr(option.find_first_not_of('-'), eq - option.find_first_not_of('-'));
			const std::string value = option.substr(eq + 1);
			const bool number = !value.empty() && value.find_first_not_of("0123456789") == std::string::npos;
			json += ", " + jsonString(key) + ": " + (number ? value : jsonString(value));
		}
		return ::tvm::Target(json + "}");
	}

	namespace {
		std::string kindOf(const std::string& spec) {
			// Unknown kinds must be reported by us (TVM would only say that it cannot parse the target).
			if (!spec.empty() && spec.front() != '{' && spec.find('/') == std::string::npos)
				return spec.substr(0, spec.find(' '));
			return parseTarget(spec)->kind->name;
		}
	} // namespace

	std::unique_ptr<Target> makeTarget(const mdl::CompileOptions& opts) {
		const std::string kind = kindOf(opts.target);
		auto it = registry().find(kind);
		if (it == registry().end()) {
			std::string known;
			for (const auto& k : targetKinds())
				known += (known.empty() ? "" : ", ") + k;
			throw std::runtime_error("target '" + kind + "' not found (available: " + known + ")");
		}
		return it->second(opts);
	}

	std::vector<std::string> targetKinds() {
		std::vector<std::string> kinds;
		for (const auto& [k, _] : registry())
			kinds.push_back(k);
		return kinds;
	}

	::tvm::IRModule
	runPasses(::tvm::IRModule mod, const std::vector<::tvm::transform::Pass>& passes, const ::tvm::Target& target) {
		auto ctx = ::tvm::transform::PassContext::Create();
		::tvm::With<::tvm::transform::PassContext> ctxGuard(ctx);
		::tvm::With<::tvm::Target> targetGuard(target);
		return ::tvm::transform::Sequential(::tvm::ffi::Array<::tvm::transform::Pass>(passes.begin(), passes.end()))(
				std::move(mod)
		);
	}

} // namespace mdl::backend::tvm
