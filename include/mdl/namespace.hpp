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

		Namespace* parent;
		std::map<std::string, std::any> objects;
		std::map<std::string, DataType> types;

		explicit Namespace(
				Namespace* parent, const std::map<std::string, std::any>& objects = {},
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
			if (it != std::end(objects)) {
				if (const T* object = std::any_cast<T>(&it->second))
					return *object;
				return std::nullopt; // an entry of another kind (e.g. a tensor whose slots are assigned one by one)
			}
			return (parent != nullptr) ? parent->get<T>(name) : std::nullopt;
		}

		/**
		 * @brief The entry named @p name in this or a parent namespace if it is a @p T (mutable, e.g. to assign a slot
		 * of a tensor); nullptr otherwise.
		 */
		template <typename T>
		T* find(std::string_view name) {
			if (auto it = objects.find(std::string{name}); it != objects.end())
				return std::any_cast<T>(&it->second);
			return parent != nullptr ? parent->find<T>(name) : nullptr;
		}

		/**
		 * @brief Rebinds the object named @p name in the closest namespace (this or a parent) that defines it.
		 * @return false if no namespace defines @p name.
		 */
		template <typename T>
		bool assign(std::string_view name, T&& obj) {
			if (auto it = objects.find(std::string{name}); it != objects.end()) {
				it->second = std::forward<T>(obj);
				return true;
			}
			return parent != nullptr && parent->assign(name, std::forward<T>(obj));
		}

		/** @brief Whether an object or type named @p name is defined in this namespace itself (not in a parent). */
		bool definesLocally(std::string_view name) const {
			return objects.contains(std::string{name}) || types.contains(std::string{name});
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
		Namespace createSubnamespace() { return Namespace{this}; }
	};

} // namespace mdl

#endif