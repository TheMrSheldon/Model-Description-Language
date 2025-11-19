#include <mdl/parser/definitions.hpp>

Computegraph empty{};

#include <type_traits>

static_assert(std::is_move_constructible_v<Program>);
static_assert(std::is_move_constructible_v<Computegraph>);
static_assert(std::is_move_constructible_v<Module>);
static_assert(std::is_move_constructible_v<Struct>);
static_assert(std::is_move_assignable_v<Program>);
static_assert(std::is_move_assignable_v<Computegraph>);
static_assert(std::is_move_assignable_v<Module>);
static_assert(std::is_move_assignable_v<Struct>);