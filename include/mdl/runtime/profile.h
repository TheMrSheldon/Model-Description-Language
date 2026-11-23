#ifndef MDL_RUNTIME_PROFILE_H
#define MDL_RUNTIME_PROFILE_H

/**
 * @file
 * @brief Profiling hooks (library `mdl_profile`): models compiled with `mdlc --profile` call these around every kernel
 * call. The times are summed per kernel kind and printed to stderr at program exit.
 */

#ifdef __cplusplus
extern "C" {
#endif

int mdl_profile_begin(const char* kernel);
int mdl_profile_end(const char* kernel);

#ifdef __cplusplus
}
#endif

#endif
