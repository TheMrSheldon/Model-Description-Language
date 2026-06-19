#include <mdl/compiler.hpp>

#include "./argvals.hpp"
#include "./backends/tvm/tvm.hpp"
#ifdef MDL_IREE_ENABLED
#include "./backends/iree/iree_backend.hpp"
#endif

void mdl::compileToFile(
		const mdl::CompilationUnit& unit,
		const std::filesystem::path& output,
		const mdl::ArgVals& argvals,
		const mdl::CompileOptions& opts
) {
#ifdef MDL_IREE_ENABLED
	if (opts.backend == mdl::Backend::Iree) {
		mdl::backends::iree::compileToFile(unit, output, argvals, opts);
		return;
	}
#endif
	mdl::backends::tvm::compileToFile(unit, output, argvals, opts);
}