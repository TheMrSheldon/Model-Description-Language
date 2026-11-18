# DLight in C++ (see README.md): TVM's default GPU schedules, ported from Python. Depends on TVM only.
add_library(mdl_dlight STATIC
	analysis/common_analysis.cpp
	base/common_schedules.cpp
	base/transform.cpp
	base/utils.cpp
	gpu/fallback.cpp
	gpu/general_reduction.cpp
	gpu/matmul.cpp
	gpu/reduction.cpp
	gpu/transpose.cpp
)
target_compile_features(mdl_dlight PUBLIC cxx_std_23)
target_link_libraries(mdl_dlight PUBLIC tvm_compiler)
# tvm_compiler uses directory-scoped include_directories internally, which do not propagate to consumers.
target_include_directories(mdl_dlight PUBLIC ${tvm_SOURCE_DIR}/include ${tvm_SOURCE_DIR}/3rdparty/tvm-ffi/include)
set_property(TARGET mdl_dlight PROPERTY POSITION_INDEPENDENT_CODE ON)
