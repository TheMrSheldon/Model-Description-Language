#include <mdl/compiler.hpp>

#include "./argvals.hpp"
#include "./backends/tvm/tvm.hpp"

void mdl::compileToFile(
		const mdl::CompilationUnit& unit, const std::filesystem::path& output, const mdl::ArgVals& argvals
) {
	mdl::backends::tvm::compileToFile(unit, output, argvals);
}