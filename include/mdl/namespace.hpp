#ifndef MDL_NAMESPACE_HPP
#define MDL_NAMESPACE_HPP

#include <any>
#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <iostream>

#include "computegraphv2/datatype.hpp"
#include "visibility.hpp"

namespace mdl {

	class Namespace final {
	private:
		/** @brief Builtin constants and functions. **/
		static const std::map<std::string, std::any> builtins;
		/** @brief Builtin datatypes. **/
		static const std::map<std::string, DataType> builtinTypes;

		Namespace* parent = nullptr;
		std::map<std::string, std::any> objects;
		std::map<std::string, DataType> types;
		/** @brief The definitions (see analysis::SemanticIndex) of the names defined here, for tools. */
		std::map<std::string, size_t> symbols;

	public:
		/**
		 * @brief How a name of this namespace is seen from other files: its visibility, and its origin (the defining
		 * file and name, the same for every import or re-export of the definition).
		 */
		struct Entry {
			Visibility visibility = Visibility::Package;
			std::string origin;
		};

	private:
		std::map<std::string, Entry> entries;

		explicit Namespace(
				Namespace* parent, const std::map<std::string, std::any>& objects = {},
				const std::map<std::string, DataType>& types = {}
		)
				: parent(parent), objects{objects}, types{types} {}

	public:
		Namespace() {}

		/** @brief Defines @p name as a copy of an object of another namespace (an import). */
		void putAny(std::string_view name, std::any obj) {
			auto [it, inserted] = objects.emplace(std::string{name}, std::move(obj));
			assert(inserted);
		}

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

		/** @brief Records the definition (an index of analysis::SemanticIndex) of @p name in this namespace. */
		void setSymbol(std::string_view name, size_t symbol) { symbols[std::string{name}] = symbol; }

		/** @brief The definition of the object or type that @p name refers to here, if recorded. */
		std::optional<size_t> symbol(std::string_view name) const {
			const std::string key{name};
			if (objects.contains(key) || types.contains(key)) {
				auto it = symbols.find(key);
				return it == symbols.end() ? std::nullopt : std::optional<size_t>{it->second};
			}
			return parent != nullptr ? parent->symbol(name) : std::nullopt;
		}

		/** @brief The definition of @p name in the enclosing namespaces (the one that a definition here hides). */
		std::optional<size_t> outerSymbol(std::string_view name) const {
			return parent != nullptr ? parent->symbol(name) : std::nullopt;
		}

		/** @brief The namespace (this or a parent) that defines @p name, if any. */
		const Namespace* definedIn(std::string_view name) const {
			if (definesLocally(name))
				return this;
			return parent != nullptr ? parent->definedIn(name) : nullptr;
		}

		/** @brief Whether an object or type named @p name is defined in this namespace itself (not in a parent). */
		bool definesLocally(std::string_view name) const {
			return objects.contains(std::string{name}) || types.contains(std::string{name});
		}

		const DataType& putType(std::string_view name, DataType&& type) {
			auto [it, inserted] = types.emplace(std::string{name}, std::move(type));
			assert(inserted);
			return it->second;
		}
		std::optional<std::reference_wrapper<const DataType>> getType(std::string_view name) const {
			auto it = types.find(std::string{name});
			if (it != std::end(types))
				return std::cref(it->second);
			if (parent != nullptr)
				return parent->getType(name);
			return std::nullopt;
		}

		/** @brief Records the visibility and origin of @p name (defined in this namespace). */
		void setEntry(std::string_view name, Entry entry) { entries[std::string{name}] = std::move(entry); }

		/** @brief The visibility and origin of @p name if it is defined in this namespace itself. */
		const Entry* entry(std::string_view name) const {
			auto it = entries.find(std::string{name});
			return it == entries.end() ? nullptr : &it->second;
		}

		/** @brief The object named @p name of this namespace itself (not a parent), any kind; nullptr if none. */
		const std::any* localObject(std::string_view name) const {
			auto it = objects.find(std::string{name});
			return it == objects.end() ? nullptr : &it->second;
		}

		/** @brief The type named @p name of this namespace itself (not a parent); nullptr if none. */
		const DataType* localType(std::string_view name) const {
			auto it = types.find(std::string{name});
			return it == types.end() ? nullptr : &it->second;
		}

		/** @brief The names of the objects and types defined in this namespace itself. */
		std::vector<std::string> localNames() const {
			std::vector<std::string> names;
			for (const auto& [name, _] : objects)
				names.push_back(name);
			for (const auto& [name, _] : types)
				if (!objects.contains(name))
					names.push_back(name);
			return names;
		}

		/** @brief The symbol of @p name if it is defined in this namespace itself. */
		std::optional<size_t> localSymbol(std::string_view name) const {
			auto it = symbols.find(std::string{name});
			return it == symbols.end() ? std::nullopt : std::optional<size_t>{it->second};
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