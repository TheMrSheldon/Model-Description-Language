#include "./planner.hpp"

#include "../tvm_compat.hpp"
#include "./rules.hpp"
#include "./signature.hpp"

#include <tvm/relax/analysis.h>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace mdl::backend::tvm::fusion {

	namespace {
		namespace relax = ::tvm::relax;
		using Kind = Signature::Kind;

		constexpr int kMaxRounds = 4;		   ///< of merging (scores change with the groups)
		constexpr int64_t kSymbolicExtent = 128; ///< the estimate of a symbolic extent (for the scores only)

		int64_t bytesOf(const ::tvm::Expr& e) {
			const auto* info = tensorInfo(e);
			const auto shape = knownShape(e);
			if (info == nullptr || !shape || !info->dtype)
				return 0;
			int64_t n = (info->dtype.value()->dtype.bits + 7) / 8;
			for (const auto& d : *shape)
				n *= d.as<::tvm::IntImmNode>() ? intValue(d.as<::tvm::IntImmNode>()) : kSymbolicExtent;
			return n;
		}

		struct Node {
			::tvm::Var var;
			Signature signature;
			std::vector<size_t> users;			 ///< nodes of the block that read the result
			std::vector<::tvm::Var> reads;		 ///< the variables it reads
			bool external = false;				 ///< the result is used after the block
		};

		/** @brief The nodes of a dataflow block, grouped by union-find. */
		class Block {
		public:
			Block(const ::tvm::IRModule& mod, const relax::BindingBlock& block, const std::unordered_set<const void*>& usedAfter) {
				std::unordered_map<const void*, size_t> index;
				for (const auto& b : block->bindings) {
					const auto* vb = b.as<relax::VarBindingNode>();
					Node n{b->var, vb ? signatureOf(mod, b->var, vb->value) : Signature{}, {}, {}, false};
					n.external = usedAfter.contains(b->var.get()) || b->var.as<relax::DataflowVarNode>() == nullptr;
					if (vb != nullptr)
						for (const auto& v : relax::FreeVars(vb->value)) {
							n.reads.push_back(v);
							if (auto it = index.find(v.get()); it != index.end())
								nodes[it->second].users.push_back(nodes.size());
						}
					index.emplace(b->var.get(), nodes.size());
					nodes.push_back(std::move(n));
				}
				parent.resize(nodes.size());
				std::iota(parent.begin(), parent.end(), 0);
				members.resize(nodes.size());
				for (size_t i = 0; i < nodes.size(); ++i)
					members[i] = {i};
			}

			Partition run() {
				const auto candidates = candidatePairs();
				for (int round = 0; round < kMaxRounds; ++round) {
					std::vector<std::tuple<int64_t, size_t, size_t, size_t>> ranked; // (-score, distance, a, b)
					for (const auto& [a, b] : candidates)
						if (find(a) != find(b))
							if (const int64_t s = score(find(a), find(b)); s > 0)
								ranked.emplace_back(-s, b - a, a, b);
					std::sort(ranked.begin(), ranked.end());
					bool merged = false;
					for (const auto& [_, distance, a, b] : ranked) {
						const size_t ga = find(a), gb = find(b);
						if (ga != gb && legal(ga, gb)) {
							unite(ga, gb);
							merged = true;
						}
					}
					if (!merged)
						break;
				}
				Partition result;
				for (size_t i = 0; i < nodes.size(); ++i)
					if (find(i) == i && members[i].size() > 1) {
						std::vector<size_t> sorted = members[i];
						std::sort(sorted.begin(), sorted.end());
						auto& vars = result.emplace_back();
						for (size_t m : sorted)
							vars.push_back(nodes[m].var);
					}
				return result;
			}

		private:
			std::vector<Node> nodes;
			std::vector<size_t> parent;
			std::vector<std::vector<size_t>> members; ///< by root

			size_t find(size_t i) {
				while (parent[i] != i)
					i = parent[i] = parent[parent[i]];
				return i;
			}

			void unite(size_t a, size_t b) {
				parent[b] = a;
				members[a].insert(members[a].end(), members[b].begin(), members[b].end());
				members[b].clear();
			}

			bool fusible(size_t i) const { return nodes[i].signature.kind != Kind::opaque; }

			/**
			 * @brief Producer → consumer edges, and each reader of a tensor with the next one (linear in the readers;
			 * merging such pairs joins all of them, as the groups grow).
			 */
			std::vector<std::pair<size_t, size_t>> candidatePairs() const {
				std::vector<std::pair<size_t, size_t>> pairs;
				std::unordered_map<const void*, size_t> lastReader;
				for (size_t i = 0; i < nodes.size(); ++i) {
					if (!fusible(i))
						continue;
					for (size_t u : nodes[i].users)
						if (fusible(u))
							pairs.emplace_back(i, u);
					for (const auto& v : nodes[i].reads) {
						auto [it, first] = lastReader.try_emplace(v.get(), i);
						if (!first) {
							pairs.emplace_back(it->second, i);
							it->second = i;
						}
					}
				}
				return pairs;
			}

			/** @brief Bytes no longer written and read (results consumed in the group), and inputs read once. */
			int64_t score(size_t a, size_t b) {
				int64_t saved = 0;
				auto edges = [&](size_t from, size_t to) {
					for (size_t m : members[from]) {
						bool feedsOther = false, usedElsewhere = nodes[m].external;
						for (size_t u : nodes[m].users) {
							const size_t g = find(u);
							feedsOther |= g == to;
							usedElsewhere |= g != to && g != from;
						}
						if (feedsOther) // (still written if used elsewhere: only the reads are saved)
							saved += bytesOf(nodes[m].var) * (usedElsewhere ? 1 : 2);
					}
				};
				edges(a, b);
				edges(b, a);
				std::unordered_set<const void*> readA;
				for (size_t m : members[a])
					for (const auto& v : nodes[m].reads)
						readA.insert(v.get());
				std::unordered_set<const void*> counted;
				for (size_t m : members[b])
					for (const auto& v : nodes[m].reads)
						if (readA.contains(v.get()) && counted.insert(v.get()).second)
							saved += bytesOf(v);
				return saved;
			}

			bool legal(size_t a, size_t b) {
				std::unordered_set<size_t> group(members[a].begin(), members[a].end());
				group.insert(members[b].begin(), members[b].end());
				// After a reduction or the anchor of the group: reachable from one inside the group.
				std::unordered_set<size_t> after;
				std::vector<size_t> work;
				for (size_t m : group)
					if (nodes[m].signature.kind == Kind::reduction || nodes[m].signature.kind == Kind::anchor)
						work.push_back(m);
				while (!work.empty()) {
					const size_t n = work.back();
					work.pop_back();
					for (size_t u : nodes[n].users)
						if (group.contains(u) && after.insert(u).second)
							work.push_back(u);
				}
				std::vector<Member> summary;
				for (size_t m : group) {
					bool escapes = nodes[m].external;
					for (size_t u : nodes[m].users)
						escapes |= !group.contains(u);
					summary.push_back({&nodes[m].signature, escapes, after.contains(m)});
				}
				return groupSignature(summary) && !cyclic(group);
			}

			/** @brief Whether a path leaves the merged group and comes back to it (it would depend on itself). */
			bool cyclic(const std::unordered_set<size_t>& group) {
				const size_t last = *std::max_element(group.begin(), group.end());
				std::unordered_set<size_t> seen;
				std::vector<size_t> work;
				for (size_t m : group)
					for (size_t u : nodes[m].users)
						if (!group.contains(u))
							work.push_back(u);
				while (!work.empty()) {
					const size_t n = work.back();
					work.pop_back();
					if (n > last || !seen.insert(n).second)
						continue;
					for (size_t u : nodes[n].users) {
						if (group.contains(u))
							return true;
						work.push_back(u);
					}
				}
				return false;
			}
		};
	} // namespace

	std::unordered_set<const void*> usedAfter(const relax::SeqExprNode& seq, size_t k) {
		std::unordered_set<const void*> used;
		for (size_t j = k + 1; j < seq.blocks.size(); ++j)
			for (const auto& b : seq.blocks[j]->bindings)
				if (const auto* vb = b.as<relax::VarBindingNode>())
					for (const auto& v : relax::FreeVars(vb->value))
						used.insert(v.get());
		for (const auto& v : relax::FreeVars(seq.body))
			used.insert(v.get());
		return used;
	}

	Partition plan(const ::tvm::IRModule& mod, const relax::Function& f) {
		const auto* seq = f->body.as<relax::SeqExprNode>();
		if (seq == nullptr)
			return {};
		Partition result;
		for (size_t k = 0; k < seq->blocks.size(); ++k) {
			if (seq->blocks[k].as<relax::DataflowBlockNode>() == nullptr)
				continue;
			for (auto& group : Block(mod, seq->blocks[k], usedAfter(*seq, k)).run())
				result.push_back(std::move(group));
		}
		return result;
	}

} // namespace mdl::backend::tvm::fusion
