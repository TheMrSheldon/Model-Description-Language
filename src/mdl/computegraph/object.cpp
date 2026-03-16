#include <mdl/computegraphv2/object.hpp>

using mdl::AggregateInstance;
using mdl::AggregateType;
using mdl::Object;

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
