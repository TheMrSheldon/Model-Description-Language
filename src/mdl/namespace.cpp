#include <mdl/namespace.hpp>

using mdl::DataType;
using mdl::Namespace;
using mdl::Primitive;

const std::map<std::string, std::any> Namespace::builtins{
		{"__builtin_cos", nullptr}, {"__builtin_sin", nullptr}, {"__builtin_tan", nullptr},
		{"__builtin_exp", nullptr}, {"__builtin_erf", nullptr},
};

const std::map<std::string, DataType> Namespace::builtinTypes{
		{"int", Primitive::integer},
		{"uint", Primitive::uint},
		{"bool", Primitive::boolean},
		{"float", Primitive::floating}
};