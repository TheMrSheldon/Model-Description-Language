#include <mdl/computegraphv2/object.hpp>

mdl::AggregateInstance::AggregateInstance(AggregateType type, std::vector<Object> members)
		: type(std::move(type)), members(std::move(members)) {}
mdl::AggregateInstance::AggregateInstance(const AggregateInstance&) = default;
mdl::AggregateInstance::AggregateInstance(AggregateInstance&&) noexcept = default;
mdl::AggregateInstance& mdl::AggregateInstance::operator=(const AggregateInstance&) = default;
mdl::AggregateInstance& mdl::AggregateInstance::operator=(AggregateInstance&&) noexcept = default;
mdl::AggregateInstance::~AggregateInstance() = default;

mdl::ModuleInstance::ModuleInstance(BoundModuleType type, std::vector<Object> params)
		: type(std::move(type)), params(std::move(params)) {}
mdl::ModuleInstance::ModuleInstance(const ModuleInstance&) = default;
mdl::ModuleInstance::ModuleInstance(ModuleInstance&&) noexcept = default;
mdl::ModuleInstance& mdl::ModuleInstance::operator=(const ModuleInstance&) = default;
mdl::ModuleInstance& mdl::ModuleInstance::operator=(ModuleInstance&&) noexcept = default;
mdl::ModuleInstance::~ModuleInstance() = default;

using mdl::AggregateInstance;
using mdl::AggregateType;
using mdl::BoundModuleType;
using mdl::ModuleInstance;
using mdl::Object;

//-----------------------
// AggregateInstance

std::expected<AggregateInstance, std::string>
AggregateInstance::create(AggregateType type, std::map<std::string, Object> values) {
	for (auto& [key, _] : values)
		if (!type.members.contains(key))
			return std::unexpected("key '" + key + "' is not a member of the type");

	for (auto& [key, _] : type.members)
		if (!values.contains(key))
			return std::unexpected("member '" + key + "' is not initialized");

	std::vector<Object> memberVec;
	memberVec.reserve(type.members.size());
	for (auto& [name, _] : type.members)
		memberVec.push_back(std::move(values.at(name)));

	return AggregateInstance{std::move(type), std::move(memberVec)};
}

std::optional<std::reference_wrapper<const Object>> AggregateInstance::getMember(std::string_view name) const {
	auto it = type.members.find(std::string(name));
	if (it == type.members.end())
		return std::nullopt;
	auto idx = std::distance(type.members.begin(), it);
	return std::cref(members[static_cast<size_t>(idx)]);
}

//-----------------------
// ModuleInstance

std::expected<ModuleInstance, std::string>
ModuleInstance::create(BoundModuleType type, std::map<std::string, Object> values) {
	/** \todo assert that BoundModuleType is fully bound */
	for (auto& [key, _] : values)
		if (type.definition.variableIdxByName(key) == ModuleType::InvalidIdx)
			return std::unexpected("key '" + key + "' is not a member of the type");

	for (auto& [key, _] : type.definition.variables)
		if (!values.contains(key))
			return std::unexpected("member '" + key + "' is not initialized");

	std::vector<Object> params;
	params.reserve(type.definition.variables.size());
	for (auto& [name, _] : type.definition.variables)
		params.push_back(std::move(values.at(name)));

	return ModuleInstance{std::move(type), std::move(params)};
}

std::optional<std::reference_wrapper<const Object>> ModuleInstance::getMember(std::string_view name) const {
	if (auto idx = type.definition.variableIdxByName(name); idx != ModuleType::InvalidIdx) {
		return std::make_optional(std::cref(params[idx]));
	}
	return std::nullopt;
}