#include <mdl/compiler.hpp>

#include "./backends/tvm/tvm.hpp"

void mdl::compileToFile(const mdl::CompilationUnit& unit, const std::filesystem::path& output) {
	mdl::backends::tvm::compileToFile(unit, output);
}