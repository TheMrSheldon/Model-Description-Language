cmake_minimum_required(VERSION 3.14...3.22)


##########################################################################################
# Package Features
##########################################################################################
include(FeatureSummary)

option(MDL_BUILD_TESTS "Build the tests" OFF)
add_feature_info(tests MDL_BUILD_TESTS "Build the MDL tests")
option(MDL_BUILD_EXAMPLES "Build the examples" OFF)
option(MDL_BENCHMARKS "Build the benchmarks of scripts/benchmark.sh (with the examples)" OFF)
add_feature_info(examples MDL_BUILD_EXAMPLES "Build the MDL examples")


##########################################################################################
# Versioning
##########################################################################################
find_package(Git)

if(GIT_EXECUTABLE)
	execute_process(
		COMMAND ${GIT_EXECUTABLE} describe --tags --dirty
		WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
		OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
		RESULT_VARIABLE GIT_DESCRIBE_ERROR_CODE
		OUTPUT_STRIP_TRAILING_WHITESPACE
    )
	if(NOT GIT_DESCRIBE_ERROR_CODE)
		string(REGEX MATCH "([0-9]+).([0-9]+).([0-9]+)(-.+)?" _ "${GIT_DESCRIBE_VERSION}")
		set(MDL_VER_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
		set(MDL_VER_MINOR ${CMAKE_MATCH_2} CACHE INTERNAL "")
		set(MDL_VER_PATCH ${CMAKE_MATCH_3} CACHE INTERNAL "")
		set(MDL_VER_META  ${CMAKE_MATCH_4} CACHE INTERNAL "")
	else()
		message(WARNING "Could not extract version information from the latest tag, ${GIT_DESCRIBE_VERSION}")
	endif()
endif()
if(NOT DEFINED MDL_VER_MAJOR OR NOT DEFINED MDL_VER_MINOR OR NOT DEFINED MDL_VER_PATCH)
	message(WARNING "I failed to read the current version from the latest git tag.")

	set(MDL_VER_MAJOR "0" CACHE INTERNAL "")
	set(MDL_VER_MINOR "0" CACHE INTERNAL "")
	set(MDL_VER_PATCH "0" CACHE INTERNAL "")
	set(MDL_VER_META  "-dev" CACHE INTERNAL "")  # https://semver.org/#spec-item-10
endif()

# Does not include the meta component since this will be used in places where CMake expects the form X.X.X
set(MDL_VERSION "${MDL_VER_MAJOR}.${MDL_VER_MINOR}.${MDL_VER_PATCH}")


##########################################################################################
# Project Setup
##########################################################################################
project(mdl VERSION ${MDL_VERSION} LANGUAGES C CXX)

# CMake >= 4 rejects dependencies that still declare cmake_minimum_required(VERSION < 3.5)
# (e.g., yaml-cpp 0.8.0). Allow them to configure with the 3.5 policy set.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# Load CPM
set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_HASH_SUM "2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a")

if(CPM_SOURCE_CACHE)
set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=${CPM_HASH_SUM}
    TLS_VERIFY OFF # Sometimes broken in the GitHub Action; we have the checksum to verify that there is nothing malicious going on; try at some point to remove this line
)

include(${CPM_DOWNLOAD_LOCATION})
message("Loaded CPM v${CURRENT_CPM_VERSION}")

# Add Source code
add_subdirectory(src)


##########################################################################################
# Examples
##########################################################################################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(MDL)
if (MDL_BUILD_TESTS OR MDL_BUILD_EXAMPLES)
	enable_testing()  # before add_subdirectory(examples) so that the examples' tests are registered
endif()
if (MDL_BUILD_EXAMPLES AND MDL_ENABLE_COMPILER)
	add_subdirectory(examples)
endif()


##########################################################################################
# Tests
##########################################################################################
if (MDL_BUILD_TESTS)
	add_subdirectory(tests)
endif()
