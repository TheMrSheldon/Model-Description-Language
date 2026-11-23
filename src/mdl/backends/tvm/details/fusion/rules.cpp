#include "./rules.hpp"

#include <algorithm>

namespace mdl::backend::tvm::fusion {

	namespace {
		using Kind = Signature::Kind;
		using Layout = Signature::Layout;

		constexpr size_t kMaxMembers = 32;
		constexpr size_t kMaxReductions = 8;

		bool sameReduction(const Signature& a, const Signature& b) {
			return a.layout == b.layout && sameDomain(a.space, b.space) && sameDomain(a.full, b.full);
		}
	} // namespace

	std::optional<Signature> groupSignature(const std::vector<Member>& members) {
		if (members.size() > kMaxMembers)
			return std::nullopt;
		// The kind: the reductions (all over the same domain) or the anchor (one) determine it.
		const Signature* reduction = nullptr;
		const Signature* anchor = nullptr;
		size_t reductions = 0;
		for (const auto& m : members) {
			const Signature& s = *m.signature;
			if (s.kind == Kind::opaque)
				return std::nullopt;
			if (s.kind == Kind::reduction) {
				if (reduction != nullptr && !sameReduction(*reduction, s))
					return std::nullopt;
				reduction = &s;
				++reductions;
			} else if (s.kind == Kind::anchor) {
				if (anchor != nullptr)
					return std::nullopt;
				anchor = &s;
			}
		}
		if (reductions > kMaxReductions || (reduction != nullptr && anchor != nullptr) ||
			std::all_of(members.begin(), members.end(), [](const Member& m) { return m.signature->view; }))
			return std::nullopt; // (views alone: copies instead of free views)

		// The results the kernel stores: of a domain that its loops cover.
		std::optional<Signature> group;
		if (reduction != nullptr) {
			group = Signature{Kind::reduction, reduction->layout, reduction->space, reduction->space, reduction->full};
		} else if (anchor != nullptr) {
			group = *anchor;
		} else {
			for (const auto& m : members)
				if (m.escapes) {
					group = Signature{Kind::pointwise, Layout::rows, m.signature->out, m.signature->out, m.signature->out};
					break;
				}
			if (!group)
				return std::nullopt;
		}
		for (const auto& m : members) {
			const Signature& s = *m.signature;
			if (s.view && m.escapes)
				return std::nullopt; // (a reshape of a result: a free view after the group, not one of its results)
			const bool full = sameDomain(s.out, group->full), perIndex = sameDomain(s.out, group->space);
			switch (group->kind) {
			case Kind::reduction:
				if (group->layout == Layout::columns) {
					// Columns: one pass over the reduced dimension; nothing of the full domain after it (a grid-wide
					// barrier), and only per-column results.
					if ((full && m.afterReduction) || (m.escapes && !perIndex))
						return std::nullopt;
				} else if (m.escapes && (!(perIndex || full) || (s.kind == Kind::reduction && !full))) {
					// Rows: results per row, or epilogues over the full rows (the reductions' results stay in the kernel).
					return std::nullopt;
				}
				break;
			case Kind::anchor:
				if (m.escapes && (!full || (s.kind == Kind::pointwise && !m.afterReduction)))
					return std::nullopt;
				break;
			default:
				if (m.escapes && !full)
					return std::nullopt;
			}
		}
		return group;
	}

} // namespace mdl::backend::tvm::fusion
