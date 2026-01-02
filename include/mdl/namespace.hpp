#ifndef MDL_NAMESPACE_HPP
#define MDL_NAMESPACE_HPP

#include <any>
#include <cassert>
#include <map>
#include <string>

#include <iostream>

#include "parser/datatype.hpp"

namespace mdl {

	class Namespace final {
	private:
		std::map<std::string, std::any> objects;
		std::map<std::string, DataType> types;

	public:
		Namespace()
				: objects{}, types{{"int", Primitive::integer},
								   {"uint", Primitive::uint},
								   {"bool", Primitive::boolean},
								   {"float", Primitive::floating}} {}

		template <typename T>
		const T& put(std::string_view name, T&& obj) {
			auto [it, inserted] = objects.emplace(std::string{name}, std::move(obj));
			assert(inserted);
			return std::any_cast<const T&>(it->second);
		}

		template <typename T>
		const T& get(std::string_view name) {
			return std::any_cast<const T&>(objects[std::string{name}]);
		}

		const DataType getType(std::string_view name) { return types[std::string{name}]; }
	};

} // namespace mdl

#endif