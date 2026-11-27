# A program that uses an installed MDL (find_package(MDL)): the sources of an example, compiled with the installed
# compiler and linked against the installed runtime. Driven by check_install.cmake.
cmake_minimum_required(VERSION 3.25)
project(mdl_install_consumer LANGUAGES CXX)

find_package(MDL REQUIRED)
message(STATUS "MDL ${MDL_VERSION}: compiler ${MDL_COMPILER}")

set(EXAMPLE_DIR "" CACHE PATH "The example to build (model.dlm, model.yml, main.cpp)")
mdl_add_model(consumer_model SOURCE "${EXAMPLE_DIR}/model.dlm" CONFIG "${EXAMPLE_DIR}/model.yml")
add_executable(consumer "${EXAMPLE_DIR}/main.cpp")
target_link_libraries(consumer PRIVATE consumer_model)
