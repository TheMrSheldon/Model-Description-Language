#include <mdl/compiler.hpp>

#include "./argvals.hpp"
#include "./backends/tvm/tvm.hpp"
#ifdef MDL_IREE_ENABLED
#include "./backends/iree/iree_backend.hpp"
#endif

#include <mdl/exports.hpp>

void mdl::compileToFile(
		const std::vector<mdl::ExportedFunction>& exports,
		const std::filesystem::path& output,
		const mdl::ArgVals& argvals,
		const mdl::CompileOptions& opts
) {
#ifdef MDL_IREE_ENABLED
	if (opts.backend == mdl::Backend::Iree) {
		mdl::backends::iree::compileToFile(exports, output, argvals, opts);
		return;
	}
#endif
	mdl::backends::tvm::compileToFile(exports, output, argvals, opts);
}
