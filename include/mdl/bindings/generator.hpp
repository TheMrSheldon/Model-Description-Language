#ifndef MDL_BINDINGS_GENERATOR_HP
#define MDL_BINDINGS_GENERATOR_HP

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace mdl {
	struct ExportedFunction;
}

namespace mdl::bindings {
	class BindingGenerator {
	public:
		virtual ~BindingGenerator() = default;

		/**
         * @brief Creates the filename for the bindings file based on the filename of the output binary.
         * @details The bindings file will typically lie next to the output binary and share a similar name. The suffix
         * is likely changed (e.g., for C++, `/path/model.o` to `/path/model.hpp`) and the filename may be updated to
         * support multiple generators into the same target language (e.g., `/path/model_pytorch.py` and
         * `/path/model_tensorflow.py`).
         */
		virtual std::filesystem::path inferBindingsFilename(const std::filesystem::path& outfile) = 0;
		/**
		 * @brief Generates bindings for the exported (compiled) functions.
		 * @see mdl::exportedFunctions for the calling convention of the compiled functions.
		 */
		virtual void generate(const std::vector<ExportedFunction>& exports, const std::filesystem::path& output) = 0;
	};

	using GeneratorFactory = std::function<std::unique_ptr<BindingGenerator>()>;
	std::map<std::string, GeneratorFactory> getGenerators();
} // namespace mdl::bindings

#endif