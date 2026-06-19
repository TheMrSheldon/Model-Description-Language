#ifndef MDL_NAMESPACE_HPP
#define MDL_NAMESPACE_HPP

#include <any>
#include <cassert>
#include <map>
#include <optional>
#include <string>

#include <iostream>

#include "computegraphv2/datatype.hpp"

namespace mdl {

	class Namespace final {
	private:
		/** @brief Builtin constants and functions. **/
		static const std::map<std::string, std::any> builtins;
		/** @brief Builtin datatypes. **/
		static const std::map<std::string, DataType> builtinTypes;

		const Namespace* parent;
		std::map<std::string, std::any> objects;
		std::map<std::string, DataType> types;

		explicit Namespace(
				const Namespace* parent, const std::map<std::string, std::any>& objects = {},
				const std::map<std::string, DataType>& types = {}
		)
				: parent(parent), objects{objects}, types{types} {}

	public:
		Namespace() {}

		template <typename T>
		const T& put(std::string_view name, T&& obj) {
			auto [it, inserted] = objects.emplace(std::string{name}, std::move(obj));
			assert(inserted);
			return std::any_cast<const T&>(it->second);
		}

		template <typename T>
		std::optional<std::reference_wrapper<const T>> get(std::string_view name) const {
			/** \todo handle misses **/
			auto it = objects.find(std::string{name});
			if (it != std::end(objects))
				return std::any_cast<const T&>(it->second);
			return (parent != nullptr) ? parent->get<T>(name) : std::nullopt;
		}

		const DataType& putType(std::string_view name, DataType&& type) {
			auto [it, inserted] = types.emplace(std::string{name}, std::move(type));
			assert(inserted);
			return std::any_cast<const DataType&>(it->second);
		}
		std::optional<std::reference_wrapper<const DataType>> getType(std::string_view name) const {
			auto it = types.find(std::string{name});
			if (it != std::end(types))
				return std::cref(it->second);
			if (parent != nullptr)
				return parent->getType(name);
			return std::nullopt;
		}

		static const Namespace emptyNamespace() { return Namespace{}; }

		static const Namespace& baseNamespace() {
			static Namespace bns{nullptr, builtins, builtinTypes};
			return bns;
		}
		Namespace createSubnamespace() const { return Namespace{this}; }
	};

} // namespace mdl

#endif