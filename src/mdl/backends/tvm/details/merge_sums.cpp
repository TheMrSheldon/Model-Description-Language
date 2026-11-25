#include "./merge_sums.hpp"

#include "../../../logging.hpp"

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	namespace {
		namespace ffi = ::tvm::ffi;
		namespace tirx = ::tvm::tirx;
		namespace s_tir = ::tvm::s_tir;
		using ::tvm::PrimExpr;

		/** @brief A loop nest of the root block: perfect loops around one block whose iterators are the loops. */
		struct Nest {
			std::vector<tirx::For> loops;
			s_tir::SBlockRealize realize;
		};

		std::optional<Nest> nestOf(const tirx::Stmt& stmt) {
			Nest n;
			tirx::Stmt body = stmt;
			while (const auto* f = body.as<tirx::ForNode>()) {
				n.loops.push_back(ffi::GetRef<tirx::For>(f));
				body = f->body;
			}
			const auto* r = body.as<s_tir::SBlockRealizeNode>();
			if (r == nullptr || r->block->iter_vars.size() != n.loops.size() || !::tvm::prim::is_one(r->predicate))
				return std::nullopt;
			for (size_t d = 0; d < n.loops.size(); ++d)
				if (!r->iter_values[d].same_as(n.loops[d]->loop_var))
					return std::nullopt;
			n.realize = ffi::GetRef<s_tir::SBlockRealize>(r);
			return n;
		}

		/** @brief Whether @p e is the constant 0 (integer or floating-point, e.g. bfloat16). */
		bool zero(const PrimExpr& e) {
			if (const auto* f = e.as<::tvm::FloatImmNode>())
				return f->value == 0;
			if (const auto* i = e.as<::tvm::IntImmNode>())
				return i->value == 0;
			return false;
		}

		/** @brief Whether the nest's block is a sum: `b = 0` initially, then `b = b + value` (one result). */
		bool isSum(const Nest& n) {
			const auto& block = n.realize->block;
			if (!block->init.has_value() || block->writes.size() != 1 || !block->alloc_buffers.empty())
				return false;
			const auto* init = block->init.value().as<tirx::BufferStoreNode>();
			const auto* store = block->body.as<tirx::BufferStoreNode>();
			if (init == nullptr || store == nullptr || !init->buffer.same_as(store->buffer) || !zero(init->value))
				return false;
			const auto* add = store->value.as<::tvm::prim::AddNode>();
			const auto* self = add ? add->a.as<::tvm::TensorLoadNode>() : nullptr;
			if (self == nullptr || !self->source.same_as(store->buffer) ||
				self->indices.size() != store->indices.size())
				return false;
			for (size_t i = 0; i < self->indices.size(); ++i)
				if (!self->indices[i].same_as(store->indices[i]))
					return false;
			return true;
		}

		/** @brief The buffers that the blocks of @p stmt read and write. */
		void accesses(
				const tirx::Stmt& stmt, std::unordered_set<const void*>& reads, std::unordered_set<const void*>& writes
		) {
			ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
					stmt,
					[&](const s_tir::SBlock& block) -> ffi::Expected<ffi::WalkResult> {
						for (const auto& r : block->reads)
							reads.insert(r->source.get());
						for (const auto& w : block->writes)
							writes.insert(w->source.get());
						return ffi::WalkResult::Advance();
					}
			);
		}

		bool intersects(const std::unordered_set<const void*>& a, const std::unordered_set<const void*>& b) {
			for (const void* x : a)
				if (b.contains(x))
					return true;
			return false;
		}

		/** @brief Whether the loops and iteration domains of @p a and @p b are the same. */
		bool sameDomain(const Nest& a, const Nest& b, ::tvm::sym::Analyzer& analyzer) {
			const auto &ia = a.realize->block->iter_vars, &ib = b.realize->block->iter_vars;
			if (a.loops.size() != b.loops.size())
				return false;
			for (size_t d = 0; d < a.loops.size(); ++d)
				if (ia[d]->iter_type != ib[d]->iter_type ||
					!analyzer->CanProveEqual(a.loops[d]->extent, b.loops[d]->extent) ||
					!analyzer->CanProveEqual(ia[d]->dom->extent, ib[d]->dom->extent))
					return false;
			return true;
		}

		/**
		 * @brief One nest computing the sums of @p a and @p b (b's iterators renamed to a's), in TVM's form of a
		 * reduction with several results.
		 */
		tirx::Stmt merged(const Nest& a, const Nest& b) {
			std::unordered_map<const void*, PrimExpr> vars;
			for (size_t d = 0; d < a.loops.size(); ++d)
				vars.emplace(b.realize->block->iter_vars[d]->var.get(), a.realize->block->iter_vars[d]->var);
			auto subst = [&](const PrimExpr& e) {
				return tirx::SubstituteWithDataTypeLegalization(e, [&](const ::tvm::Var& v) -> ffi::Optional<PrimExpr> {
					auto it = vars.find(v.get());
					return it == vars.end() ? ffi::Optional<PrimExpr>() : ffi::Optional<PrimExpr>(it->second);
				});
			};
			auto store = [&](const tirx::Stmt& s) -> tirx::Stmt {
				const auto* st = s.as<tirx::BufferStoreNode>();
				ffi::Array<PrimExpr> indices;
				for (const auto& i : st->indices)
					indices.push_back(subst(i));
				return tirx::BufferStore(st->buffer, subst(st->value), indices);
			};
			auto regions = [&](const ffi::Array<::tvm::TensorRegion>& rs, ffi::Array<::tvm::TensorRegion> into) {
				for (const auto& r : rs) {
					ffi::Array<::tvm::Range> region;
					for (const auto& range : r->region)
						region.push_back(::tvm::Range::FromMinExtent(subst(range->min), subst(range->extent)));
					into.push_back(tirx::BufferRegion(r->source.as_or_throw<tirx::BufferVar>(), region));
				}
				return into;
			};
			const auto& bb = b.realize->block;
			auto block = a.realize->block;
			const auto* s0 = block->body.as<tirx::BufferStoreNode>();
			const auto second = store(bb->body);
			const auto* s1 = second.as<tirx::BufferStoreNode>();
			auto* n = block.CopyOnWrite();
			n->init = tirx::SeqStmt({block->init.value(), store(bb->init.value())});
			// (TVM's form of a reduction with several results: the new values bound first, then stored)
			const ::tvm::PrimVar v0("v0", s0->value->ty.as_or_throw<::tvm::PrimType>()),
					v1("v1", s1->value->ty.as_or_throw<::tvm::PrimType>());
			n->body = tirx::SeqStmt(
					{tirx::Bind(v0, s0->value), tirx::Bind(v1, s1->value),
					 tirx::BufferStore(s0->buffer, v0, s0->indices), tirx::BufferStore(s1->buffer, v1, s1->indices)}
			);
			n->reads = regions(bb->reads, block->reads);
			n->writes = regions(bb->writes, block->writes);
			auto realize = a.realize;
			realize.CopyOnWrite()->block = block;
			tirx::Stmt body = realize;
			for (size_t d = a.loops.size(); d-- > 0;) {
				auto loop = a.loops[d];
				loop.CopyOnWrite()->body = body;
				body = loop;
			}
			return body;
		}

		std::optional<tirx::PrimFunc> mergeSums(const tirx::PrimFunc& func) {
			const auto* rootRealize = func->body.as<s_tir::SBlockRealizeNode>();
			const auto* seq = rootRealize ? rootRealize->block->body.as<tirx::SeqStmtNode>() : nullptr;
			if (seq == nullptr)
				return std::nullopt;
			// The statements of the root block, their buffer accesses, and their order constraints (a statement after
			// another that writes what it reads or writes, or that reads what it writes).
			struct Node {
				tirx::Stmt stmt;
				std::unordered_set<const void*> reads, writes;
			};
			std::vector<Node> nodes;
			for (const auto& stmt : seq->seq) {
				Node n{stmt, {}, {}};
				accesses(stmt, n.reads, n.writes);
				nodes.push_back(std::move(n));
			}
			const size_t count = nodes.size();
			// before[u][v]: statement u must stay before v (v writes what u reads or writes, or reads what u writes).
			std::vector<std::vector<bool>> before(count, std::vector<bool>(count, false));
			for (size_t v = 0; v < count; ++v)
				for (size_t u = 0; u < v; ++u)
					before[u][v] = intersects(nodes[v].reads, nodes[u].writes) ||
								   intersects(nodes[v].writes, nodes[u].writes) ||
								   intersects(nodes[v].writes, nodes[u].reads);
			// Transitively.
			for (size_t m = 0; m < count; ++m)
				for (size_t u = 0; u < count; ++u)
					if (before[u][m])
						for (size_t v = 0; v < count; ++v)
							if (before[m][v])
								before[u][v] = true;
			// The groups of original statements (a merged pair: two), in the order of their first statement.
			std::vector<std::vector<size_t>> groups;
			for (size_t k = 0; k < count; ++k)
				groups.push_back({k});
			::tvm::sym::Analyzer analyzer;
			bool changed = false;
			std::vector<bool> taken(count, false);
			// (pairs: TVM's reducers have at most two results)
			for (size_t i = 0; i < count; ++i) {
				const auto a = nestOf(nodes[i].stmt);
				if (taken[i] || !a || !isSum(*a))
					continue;
				for (size_t j = i + 1; j < count; ++j) {
					const auto b = nestOf(nodes[j].stmt);
					if (taken[j] || !b || !isSum(*b) || before[i][j] || !sameDomain(*a, *b, analyzer))
						continue;
					nodes[i].stmt = merged(*a, *b);
					taken[i] = taken[j] = true;
					groups[i].push_back(j);
					groups[j].clear();
					changed = true;
					break;
				}
			}
			if (!changed)
				return std::nullopt;
			std::erase_if(groups, [](const auto& g) { return g.empty(); });
			// The groups in an order that respects the constraints (the earliest available first). (A merged pair has
			// no path between its statements, so the constraints between the groups have no cycle.)
			auto mustPrecede = [&](const std::vector<size_t>& x, const std::vector<size_t>& y) {
				for (size_t u : x)
					for (size_t v : y)
						if (before[u][v])
							return true;
				return false;
			};
			std::vector<tirx::Stmt> stmts;
			std::vector<bool> done(groups.size(), false);
			for (size_t emitted = 0; emitted < groups.size(); ++emitted) {
				std::optional<size_t> next;
				for (size_t g = 0; g < groups.size() && !next; ++g) {
					if (done[g])
						continue;
					bool ready = true;
					for (size_t h = 0; h < groups.size() && ready; ++h)
						ready = done[h] || h == g || !mustPrecede(groups[h], groups[g]);
					if (ready)
						next = g;
				}
				if (!next)
					return std::nullopt; // (not expected)
				done[*next] = true;
				stmts.push_back(nodes[groups[*next][0]].stmt);
			}
			auto root = rootRealize->block;
			root.CopyOnWrite()->body =
					stmts.size() == 1 ? stmts[0] : tirx::SeqStmt(ffi::Array<tirx::Stmt>(stmts.begin(), stmts.end()));
			auto rr = ffi::GetRef<s_tir::SBlockRealize>(rootRealize);
			rr.CopyOnWrite()->block = root;
			auto result = func;
			result.CopyOnWrite()->body = rr;
			return result;
		}
	} // namespace

	::tvm::IRModule mergeSiblingSums(::tvm::IRModule kernels) {
		std::vector<std::pair<::tvm::GlobalVar, tirx::PrimFunc>> updates;
		for (const auto& [gv, func] : kernels->functions)
			if (const auto* prim = func.as<tirx::PrimFuncNode>())
				if (auto merged = mergeSums(ffi::GetRef<tirx::PrimFunc>(prim)))
					updates.emplace_back(gv, *merged);
		for (auto& [gv, func] : updates)
			kernels->Update(gv, func);
		mdl::log::getLogger("compiler")->info("Merged sibling sums in {} kernel(s)", updates.size());
		return kernels;
	}

} // namespace mdl::backend::tvm
