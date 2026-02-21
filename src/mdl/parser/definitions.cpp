#include <mdl/parser/definitions.hpp>

mdl::Computegraph empty{};

#include <type_traits>

static_assert(std::is_move_constructible_v<mdl::Program>);
static_assert(std::is_move_constructible_v<mdl::Computegraph>);
static_assert(std::is_move_constructible_v<mdl::Module>);
static_assert(std::is_move_constructible_v<mdl::Struct>);
static_assert(std::is_move_assignable_v<mdl::Program>);
static_assert(std::is_move_assignable_v<mdl::Computegraph>);
static_assert(std::is_move_assignable_v<mdl::Module>);
static_assert(std::is_move_assignable_v<mdl::Struct>);