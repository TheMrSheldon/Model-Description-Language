// Placeholder translation unit for builds without vendor kernels (MDL_ENABLE_ONEDNN=OFF). Models compiled for such a
// build use TVM-generated kernels only (`mdlc --kernels none`) and do not reference any symbol of mdl_kernels.
#include <mdl/runtime/kernels.h>
