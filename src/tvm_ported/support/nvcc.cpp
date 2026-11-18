// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/support/nvcc.py
// (conventions: src/tvm_ported/README.md)
#include "./nvcc.hpp"

#include <tvm/target/target.h>

#include <nvrtc.h>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <sys/utsname.h>

namespace mdl::tvm_ported::support::nvcc {

	namespace {
		namespace fs = std::filesystem;

		std::string env(const char* name, const std::string& fallback = "") {
			const char* value = std::getenv(name);
			return value != nullptr ? value : fallback;
		}

		/** @brief Python's shlex.split, for whitespace-separated flags (no quoting). */
		std::vector<std::string> split(const std::string& s) {
			std::istringstream in(s);
			std::vector<std::string> result;
			for (std::string token; in >> token;)
				result.push_back(token);
			return result;
		}

		bool starts_with(const std::string& s, const std::string& prefix) { return s.starts_with(prefix); }

		/** @brief Return ptxas flags forwarded via `--ptxas-options` (without the prefix). */
		std::vector<std::string> _ptxas_option_flags() {
			std::vector<std::string> flags{
					"-v",
					"--register-usage-level=" + env("TVM_CUDA_PTXAS_REG_LEVEL", "10"),
					"--warn-on-local-memory-usage",
			};
			for (auto& f : split(env("TVM_CUDA_PTXAS_EXTRA_OPTS")))
				flags.push_back(f);
			return flags;
		}

		/** @brief Find the architecture-specific `targets/<triple>/include` directory. */
		std::optional<std::string> _find_cuda_target_include(const std::string& cuda_path) {
			utsname u{};
			uname(&u);
			const std::string machine = u.machine, system = "linux";
			std::vector<std::string> triples;
			if (machine == "aarch64" || machine == "arm64")
				triples = {"sbsa-linux", machine + "-" + system};
			else
				triples = {machine + "-" + system};
			for (const auto& triple : triples) {
				const fs::path include_dir = fs::path(cuda_path) / "targets" / triple / "include";
				if (fs::is_directory(include_dir))
					return include_dir.string();
			}
			return std::nullopt;
		}

		void check(nvrtcResult result, nvrtcProgram* prog, const std::string& what) {
			if (result == NVRTC_SUCCESS)
				return;
			if (prog != nullptr)
				nvrtcDestroyProgram(prog);
			throw std::runtime_error(what + ": " + nvrtcGetErrorString(result));
		}

		/** @brief Compile CUDA code using NVRTC (NVIDIA Runtime Compilation). */
		std::string _compile_cuda_nvrtc(
				const std::string& code, std::optional<std::string> target_format, std::optional<std::string> arch,
				const std::vector<std::string>& options
		) {
			if (!target_format)
				target_format = "cubin";
			if (*target_format == "fatbin")
				throw std::invalid_argument(
						"NVRTC does not support fatbin generation yet. Use target_format='cubin' or 'ptx' with NVRTC."
				);
			if (*target_format != "cubin" && *target_format != "ptx")
				throw std::invalid_argument("target_format must be 'cubin' or 'ptx', got: " + *target_format);

			// NVRTC selects the output kind through both target_format and the architecture spelling. A virtual
			// ``compute_*`` target produces PTX, while a real ``sm_*`` target produces a loadable cubin.
			if (!arch) {
				auto target = ::tvm::Target::Current(/*allow_not_defined=*/true);
				auto a = target.defined() ? target->GetAttr<::tvm::ffi::String>("arch") : std::nullopt;
				if (!a)
					throw std::runtime_error("compile_cuda: no CUDA architecture (the current target has no arch)");
				arch = std::string(a.value());
			}
			if (*target_format == "ptx" && starts_with(*arch, "sm_"))
				arch = "compute_" + arch->substr(3);
			else if (*target_format == "cubin" && starts_with(*arch, "compute_"))
				arch = "sm_" + arch->substr(8);

			// Strip host-only headers for NVRTC. NVRTC compiles device code and does not require the CUDA driver
			// header or host C++ headers.
			std::istringstream lines(code);
			std::string code_filtered;
			for (std::string line; std::getline(lines, line);) {
				const auto first = line.find_first_not_of(" \t"), last = line.find_last_not_of(" \t\r");
				const std::string stripped = first == std::string::npos ? "" : line.substr(first, last - first + 1);
				if (stripped == "#include <cuda.h>")
					continue;
				code_filtered += (code_filtered.empty() ? "" : "\n") + line;
			}

			// NVRTC does not include the host-side cuda.h, so the complete ``CUtensorMap_st`` layout is missing.
			if (code_filtered.find("CUtensorMap") != std::string::npos)
				code_filtered = "struct alignas(64) CUtensorMap_st {\n"
								"  unsigned long long opaque[16];\n"
								"};\n"
								"typedef struct CUtensorMap_st CUtensorMap;\n\n" +
								code_filtered;

			// Add standard type definitions and compatibility macros that NVRTC doesn't provide.
			const std::string nvrtc_preamble = R"TVM(#include <cuda/std/cstdint>
using cuda::std::uint8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
using cuda::std::int8_t;
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;

// NVRTC uses asm/volatile instead of __asm__/__volatile__
#ifndef __asm__
#define __asm__ asm
#endif
#ifndef __volatile__
#define __volatile__ volatile
#endif

// NVRTC does not pull in the host <math.h>, so INFINITY is undefined. Provide it
// from libcu++ (same float +inf value nvcc's <math.h> yields).
#include <cuda/std/limits>
#ifndef INFINITY
#define INFINITY (::cuda::std::numeric_limits<float>::infinity())
#endif

)TVM";
			code_filtered = nvrtc_preamble + code_filtered;

			nvrtcProgram prog = nullptr;
			check(nvrtcCreateProgram(&prog, code_filtered.c_str(), "tvm_kernels.cu", 0, nullptr, nullptr), nullptr,
				  "Failed to create NVRTC program");

			// Prepare compilation options
			const std::string cuda_path = find_cuda_path();
			std::vector<std::string> compile_opts{
					"--gpu-architecture=" + *arch,
					"-default-device",
					// nvcc enables 128-bit integers by default on Linux; NVRTC requires the flag to be passed
					// explicitly for kernels that use __int128_t.
					"--device-int128",
			};

			// Add CUDA include paths. NVRTC needs explicit include paths for CUDA headers.
			std::vector<std::string> include_paths;
			const fs::path standard_include = fs::path(cuda_path) / "include";
			if (fs::is_directory(standard_include))
				include_paths.push_back(standard_include.string());
			const auto arch_include = _find_cuda_target_include(cuda_path);
			if (arch_include)
				include_paths.push_back(*arch_include);
			// Check for CCCL include directory (required for cuda/std/cstdint and type_traits)
			if (arch_include && fs::is_directory(fs::path(*arch_include) / "cccl"))
				include_paths.push_back((fs::path(*arch_include) / "cccl").string());
			bool found = false;
			for (const auto& p : include_paths)
				found |= fs::is_regular_file(fs::path(p) / "cuda_runtime.h");
			if (!found) {
				nvrtcDestroyProgram(&prog);
				throw std::runtime_error("Cannot find CUDA headers (cuda_runtime.h) under " + cuda_path);
			}
			for (const auto& include_path : include_paths)
				compile_opts.push_back("-I" + include_path);

			for (const char* opt :
				 {"-D__NV_SILENCE_DEPRECATION_BEGIN=", "-D__NV_SILENCE_DEPRECATION_END=",
				  "-D__NV_SILENCE_HOST_DEPRECATION_BEGIN=", "-D__NV_SILENCE_HOST_DEPRECATION_END=",
				  "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
				  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
				  "-U__CUDA_NO_BFLOAT162_CONVERSIONS__", "--use_fast_math"})
				compile_opts.emplace_back(opt);

			for (const auto& flag : _ptxas_option_flags())
				compile_opts.push_back("--ptxas-options=" + flag);

			for (auto& t : split(env("TVM_CUDA_NVRTC_EXTRA_OPTS")))
				compile_opts.push_back(t);

			// Add user-provided options, filtering out nvcc-specific flags that nvrtc doesn't support
			static const std::vector<std::string> nvcc_only_prefixes{
					"-c", "-O", "-std", "--std", "-Xcompiler", "-Xlinker", "-Xarchive", "-Xcudafe", "-Xptxas",
					"--compile", "--compiler-options", "--linker-options", "-fPIC", "-shared", "-o",
			};
			for (const auto& opt : options) {
				bool skip = false;
				for (const auto& prefix : nvcc_only_prefixes)
					skip |= starts_with(opt, prefix);
				if (!skip)
					compile_opts.push_back(opt);
			}

			// Compile
			std::vector<const char*> argv;
			for (const auto& o : compile_opts)
				argv.push_back(o.c_str());
			const nvrtcResult result = nvrtcCompileProgram(prog, static_cast<int>(argv.size()), argv.data());
			if (result != NVRTC_SUCCESS) {
				// Get compilation log
				size_t log_size = 0;
				std::string error_msg;
				if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS && log_size > 0) {
					std::string log_buf(log_size, '\0');
					if (nvrtcGetProgramLog(prog, log_buf.data()) == NVRTC_SUCCESS)
						error_msg = "NVRTC compilation failed:\n" + log_buf;
					else
						error_msg = std::string("NVRTC compilation failed (couldn't get log): ") +
									nvrtcGetErrorString(result);
				} else {
					error_msg = std::string("NVRTC compilation failed: ") + nvrtcGetErrorString(result);
				}
				nvrtcDestroyProgram(&prog);
				throw std::runtime_error(error_msg);
			}

			// Get compiled binary
			std::string binary_buf;
			size_t binary_size = 0;
			if (*target_format == "cubin") {
				check(nvrtcGetCUBINSize(prog, &binary_size), &prog, "Failed to get CUBIN size");
				binary_buf.resize(binary_size);
				check(nvrtcGetCUBIN(prog, binary_buf.data()), &prog, "Failed to get CUBIN");
			} else { // ptx
				check(nvrtcGetPTXSize(prog, &binary_size), &prog, "Failed to get PTX size");
				binary_buf.resize(binary_size);
				check(nvrtcGetPTX(prog, binary_buf.data()), &prog, "Failed to get PTX");
			}

			// Clean up NVRTC program
			nvrtcDestroyProgram(&prog);
			return binary_buf;
		}
	} // namespace

	std::string compile_cuda(
			const std::string& code, std::optional<std::string> target_format, std::optional<std::string> arch,
			const std::vector<std::string>& options
	) {
		// (Not ported: the nvcc compiler and NVSHMEM.)
		return _compile_cuda_nvrtc(code, std::move(target_format), std::move(arch), options);
	}

	std::string find_cuda_path() {
		if (const char* path = std::getenv("CUDA_PATH"))
			return path;
		// `which nvcc`
		const std::string path_env = env("PATH");
		std::istringstream dirs(path_env);
		for (std::string dir; std::getline(dirs, dir, ':');) {
			const fs::path nvcc = fs::path(dir) / "nvcc";
			if (!dir.empty() && fs::is_regular_file(nvcc))
				return fs::weakly_canonical(nvcc / ".." / "..").string();
		}
		const std::string cuda_path = "/usr/local/cuda";
		if (fs::exists(fs::path(cuda_path) / "bin/nvcc"))
			return cuda_path;
		throw std::runtime_error("Cannot find CUDA path");
	}

	std::string tvm_callback_cuda_compile(const std::string& code) {
		// The current Target is fetched inside compile_cuda when arch is unset.
		std::string compiler = env("TVM_CUDA_COMPILE_MODE", "nvrtc");
		for (auto& c : compiler)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		if (compiler == "nvrtc")
			return compile_cuda(code, "cubin", std::nullopt, {});
		throw std::invalid_argument("TVM_CUDA_COMPILE_MODE: only nvrtc is supported (got " + compiler + ")");
	}

} // namespace mdl::tvm_ported::support::nvcc
