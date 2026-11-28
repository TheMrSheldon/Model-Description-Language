#include "./flatten_injective.hpp"

#include "../../../logging.hpp"

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/attrs.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <exception>
#include <map>
#include <optional>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	namespace {
		namespace ffi = ::tvm::ffi;
		namespace tirx = ::tvm::tirx;
		namespace s_tir = ::tvm::s_tir;
		using ::tvm::PrimExpr;
		using ::tvm::PrimVar;
		using DataType = ::tvm::PrimType;

		/** @brief The row-major offset of @p indices in a buffer of @p shape. */
		PrimExpr offsetOf(const ffi::Array<PrimExpr>& indices, const ffi::Array<PrimExpr>& shape) {
			PrimExpr offset = ::tvm::IntImm(DataType::Int(64), 0);
			for (size_t i = 0; i < indices.size(); ++i)
				offset = offset * ::tvm::prim::cast(DataType::Int(64), shape[i]) +
						 ::tvm::prim::cast(DataType::Int(64), indices[i]);
			return offset;
		}

		/** @brief A compact (dense, row-major, unshifted) buffer, which a flat view can replace. */
		bool compact(const tirx::BufferVar& buffer) {
			return buffer->strides.empty() && ::tvm::prim::is_zero(buffer->elem_offset);
		}

		/** @brief A 1-D view of @p elements elements over the memory of @p buffer. */
		tirx::BufferVar flatView(const tirx::BufferVar& buffer, const PrimExpr& elements) {
			return tirx::BufferVar(
					buffer.name() + "_flat",
					tirx::BufferType(
							buffer->storage_scope, buffer->dtype, {elements}, {}, ::tvm::IntImm(DataType::Int(64), 0),
							buffer->data_alignment, buffer->offset_factor
					)
			);
		}

		std::optional<tirx::PrimFunc> flatten(const tirx::PrimFunc& func) {
			// One block after inlining the elementwise producers (the kernels of TE computations: a root block whose
			// children are the operations).
			s_tir::Schedule sch = s_tir::Schedule::Concrete(
					::tvm::IRModule({{::tvm::GlobalVar("main"), func}}), /*seed=*/0, /*debug_mask=*/0,
					s_tir::ScheduleErrorRenderLevel::kNone
			);
			const auto* realize = func->body.as<s_tir::SBlockRealizeNode>();
			if (realize == nullptr)
				return std::nullopt;
			const s_tir::SBlockRV root = sch->GetSBlock(realize->block->name_hint, ffi::String("main"));
			for (bool progress = true; progress;) {
				progress = false;
				const auto children = sch->GetChildBlocks(root);
				if (children.size() <= 1)
					break;
				for (const auto& block : children) {
					try {
						sch->ComputeInline(block);
						progress = true;
						break;
					} catch (const std::exception&) {
					}
				}
			}
			const auto children = sch->GetChildBlocks(root);
			if (children.size() != 1)
				return std::nullopt;
			const s_tir::SBlock block = sch->Get(children[0]);
			const auto loops = sch->GetLoops(children[0]);
			if (block->init.has_value() || !block->match_buffers.empty() || !block->alloc_buffers.empty() ||
				loops.size() != block->iter_vars.size() || block->iter_vars.empty())
				return std::nullopt;
			for (const auto& iv : block->iter_vars)
				if (iv->iter_type != tirx::kDataPar || !::tvm::prim::is_zero(iv->dom->min))
					return std::nullopt;
			const auto rootBlock = sch->Get(root);
			if (!rootBlock->alloc_buffers.empty() || block->iter_vars.size() < 2)
				return std::nullopt; // (a single loop is already flat)

			// The results (several: merged sibling kernels, fuseSiblings): each written at the iterators, in order.
			std::vector<const tirx::BufferStoreNode*> stores;
			if (const auto* seq = block->body.as<tirx::SeqStmtNode>()) {
				for (const auto& s : seq->seq)
					stores.push_back(s.as<tirx::BufferStoreNode>());
			} else {
				stores.push_back(block->body.as<tirx::BufferStoreNode>());
			}
			for (const auto* store : stores) {
				if (store == nullptr || !compact(store->buffer) || store->indices.size() != block->iter_vars.size())
					return std::nullopt;
				for (size_t i = 0; i < store->indices.size(); ++i)
					if (!store->indices[i].same_as(block->iter_vars[i]->var))
						return std::nullopt;
			}
			ffi::Array<PrimExpr> extents;
			PrimExpr total = ::tvm::IntImm(DataType::Int(64), 1);
			for (const auto& iv : block->iter_vars) {
				extents.push_back(iv->dom->extent);
				total = total * ::tvm::prim::cast(DataType::Int(64), iv->dom->extent);
			}
			ffi::Array<PrimExpr> vars;
			for (const auto& iv : block->iter_vars)
				vars.push_back(iv->var);

			::tvm::sym::Analyzer analyzer;
			for (const auto& iv : block->iter_vars)
				analyzer->Bind(iv->var.as_or_throw<PrimVar>(), iv->dom);
			for (const auto& e : extents)
				for (const auto& v : tirx::UndefinedVars(e))
					analyzer->MarkGlobalNonNegValue(v.as_or_throw<PrimVar>());
			const PrimExpr resultOffset = offsetOf(vars, extents);
			// The trailing dimensions: suffix s covers iterators [s, ndim), of static size P.
			std::vector<std::pair<PrimExpr, int64_t>> suffixes; // (offset of the suffix, P)
			{
				int64_t P = 1;
				for (size_t s = extents.size(); s-- > 1;) {
					const auto* imm = extents[s].as<::tvm::IntImmNode>();
					if (imm == nullptr)
						break;
					P *= static_cast<int64_t>(imm->value);
					ffi::Array<PrimExpr> trailingVars(vars.begin() + static_cast<std::ptrdiff_t>(s), vars.end());
					ffi::Array<PrimExpr> trailingExtents(
							extents.begin() + static_cast<std::ptrdiff_t>(s), extents.end()
					);
					suffixes.emplace_back(offsetOf(trailingVars, trailingExtents), P);
				}
			}

			// The operands: each load becomes a load of a flat view at n (the result's layout) or n % P (a broadcast of
			// trailing dimensions); loads that do not depend on the iterators stay.
			PrimVar n("vn", DataType::Int(64));			  // the block iterator of the flat loop
			std::map<const void*, tirx::BufferVar> views; // by buffer
			std::vector<std::pair<tirx::BufferVar, tirx::BufferVar>> declared;
			auto viewOf = [&](const tirx::BufferVar& buffer, const PrimExpr& elements) {
				auto it = views.find(buffer.get());
				if (it != views.end())
					return it->second;
				auto view = flatView(buffer, elements);
				views.emplace(buffer.get(), view);
				declared.emplace_back(buffer, view);
				return view;
			};
			std::unordered_set<const void*> iterators;
			for (const auto& v : vars)
				iterators.insert(v.get());
			auto usesIterators = [&](const PrimExpr& e) {
				for (const auto& v : tirx::UndefinedVars(e))
					if (iterators.contains(v.get()))
						return true;
				return false;
			};

			bool ok = true;
			ffi::Array<::tvm::TensorRegion> reads, writes;
			ffi::Array<tirx::Stmt> flatStores;
			for (const auto* store : stores) {
				PrimExpr value = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
										 store->value,
										 [&](const ::tvm::TensorLoad& load) -> ffi::UnchangedOr<ffi::Any> {
											 auto buffer = load->source.as<tirx::BufferVar>();
											 if (!buffer || !ok)
												 return ffi::Unchanged();
											 bool dependent = false;
											 for (const auto& i : load->indices)
												 dependent |= usesIterators(i);
											 if (!dependent) {
												 reads.push_back(tirx::BufferRegionFromPoint(*buffer, load->indices));
												 return ffi::Unchanged();
											 }
											 if (!compact(*buffer)) {
												 ok = false;
												 return ffi::Unchanged();
											 }
											 const PrimExpr offset = offsetOf(load->indices, (*buffer)->shape);
											 PrimExpr elements = ::tvm::IntImm(DataType::Int(64), 1);
											 for (const auto& d : (*buffer)->shape)
												 elements = elements * ::tvm::prim::cast(DataType::Int(64), d);
											 std::optional<PrimExpr> index;
											 if (analyzer->CanProveEqual(offset, resultOffset))
												 index = n;
											 for (const auto& [suffixOffset, P] : suffixes)
												 if (!index && analyzer->CanProveEqual(offset, suffixOffset))
													 index = ::tvm::floormod(n, ::tvm::IntImm(DataType::Int(64), P));
											 if (!index) {
												 ok = false;
												 return ffi::Unchanged();
											 }
											 const auto view = viewOf(*buffer, elements);
											 reads.push_back(tirx::BufferRegionFromPoint(view, {*index}));
											 return ffi::Any(tirx::BufferLoad(view, {*index}));
										 }
				).cast<PrimExpr>();
				if (!ok || usesIterators(value))
					return std::nullopt;
				// flat[n] = value
				const auto out = viewOf(store->buffer, total);
				writes.push_back(tirx::BufferRegionFromPoint(out, {n}));
				flatStores.push_back(tirx::BufferStore(out, value, {n}));
			}

			// for n in range(total): the stores
			const PrimVar& vn = n;
			const ffi::Array<::tvm::TensorRegion>& blockReads = reads;
			const s_tir::SBlock flat(
					{tirx::IterVar(
							::tvm::Range::FromMinExtent(::tvm::IntImm(DataType::Int(64), 0), total), vn, tirx::kDataPar
					)},
					blockReads, writes, block->name_hint,
					flatStores.size() == 1 ? flatStores[0] : tirx::Stmt(tirx::SeqStmt(flatStores))
			);
			PrimVar loop("i", DataType::Int(64));
			std::vector<tirx::Stmt> body;
			for (const auto& [buffer, view] : declared)
				body.push_back(tirx::DeclBuffer(view, buffer.data()));
			body.push_back(tirx::For(
					loop, ::tvm::IntImm(DataType::Int(64), 0), total, tirx::ForKind::kSerial,
					s_tir::SBlockRealize({loop}, ::tvm::IntImm::Bool(true), flat)
			));
			const s_tir::SBlock newRoot({}, {}, {}, realize->block->name_hint, tirx::SeqStmt::Flatten(body));
			auto result = func;
			result.CopyOnWrite()->body = s_tir::SBlockRealize({}, ::tvm::IntImm::Bool(true), newRoot);
			return result;
		}
	} // namespace

	::tvm::IRModule flattenInjective(::tvm::IRModule kernels, const ::tvm::Target& /*target*/) {
		auto logger = mdl::log::getLogger("compiler");
		std::vector<std::pair<::tvm::GlobalVar, tirx::PrimFunc>> updates;
		for (const auto& [gv, func] : kernels->functions) {
			const auto* prim = func.as<tirx::PrimFuncNode>();
			if (prim == nullptr)
				continue;
			try {
				if (auto flat = flatten(ffi::GetRef<tirx::PrimFunc>(prim)))
					updates.emplace_back(gv, *flat);
			} catch (const std::exception& e) {
				logger->debug("flattenInjective skipped {}: {}", std::string(gv->name_hint), e.what());
			}
		}
		for (auto& [gv, func] : updates)
			kernels->Update(gv, func);
		logger->info("Flattened {} injective kernel(s)", updates.size());
		return kernels;
	}

} // namespace mdl::backend::tvm
