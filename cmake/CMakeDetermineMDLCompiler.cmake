find_program(
    CMAKE_MDL_COMPILER
        NAMES "mdlc"
        HINTS ${CMAKE_BINARY_DIR}/src
        DOC "Compiler for the model description language (mdl)"
)

set(CMAKE_MDL_SOURCE_FILE_EXTENSIONS mdl;dlm)
set(CMAKE_MDL_OUTPUT_EXTENSION ${CMAKE_C_OUTPUT_EXTENSION})
set(CMAKE_MDL_COMPILER_ENV_VAR "")



# set(CMAKE_MDL_COMPILE_OBJECT
#     "<CMAKE_COMMAND> -E make_directory <OBJECT_DIR>"
#     "&& ${MDLC_EXECUTABLE} <SOURCE>"
#     "-o <OBJECT>"
#     "-h <OBJECT_DIR>/<SOURCE_BASE>.h"
# )

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeMDLCompiler.cmake.in ${CMAKE_PLATFORM_INFO_DIR}/CMakeMDLCompiler.cmake)