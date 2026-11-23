#include "./signature.hpp"

#include "../tvm_compat.hpp"

#include <tvm_ported/s_tir/dlight/analysis/common_analysis.hpp>
#include <tvm_ported/s_tir/dlight/base/common_schedules.hpp>
#include <tvm_ported/s_tir/dlight/base/utils.hpp>

#include <tvm/ir/attrs.h>
#include <tvm/relax/analysis.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>

#include <exception>
#include <unordered_set>

namespace mdl::backend::tvm::fusion {

	namespace {
		namespace dl = mdl::tvm_ported::dlight;
		using Kind = Signature::Kind;
		using Layout = Signature::Layout;

		std::optional<Domain> domainOf(const ::tvm::Expr& e) {
			const auto shape = knownShape(e);
			if (!shape)
				return std::nullopt;
			Domain d;
			for (const auto& extent : *shape)
				if (!::tvm::prim::is_one(extent))
					d.push_back(extent);
			return d;
		}

		bool hasAttr(const ::tvm::tirx::PrimFunc& f, const char* name, int64_t value) {
			auto attr = f->GetAttr<::tvm::IntImm>(name);
			return attr && intValue(attr.value().get()) == value;
		}

		/** @brief The layout of the reductions: of which dimension the reduced reads' innermost index is (none: mixed). */
		std::optional<Layout> reductionLayout(const ::tvm::s_tir::Schedule& sch, const std::vector<dl::analysis::SBlockInfo>& infos) {
			bool rows = false, columns = false;
			for (const auto& info : infos) {
				if (!info.is_reduction())
					continue;
				const auto block = sch->Get(info.block_rv);
				std::unordered_set<const void*> reduced;
				for (const auto& iv : block->iter_vars)
					if (iv->iter_type == ::tvm::tirx::kCommReduce)
						reduced.insert(iv->var.get());
				auto uses = [&](const ::tvm::PrimExpr& e) {
					for (const auto& v : ::tvm::tirx::UndefinedVars(e))
						if (reduced.contains(v.get()))
							return true;
					return false;
				};
				for (const auto& read : block->reads) {
					bool dependent = false;
					for (const auto& range : read->region)
						dependent |= uses(range->min);
					if (!dependent || read->region.empty())
						continue;
					(uses(read->region.back()->min) ? rows : columns) = true;
				}
			}
			if (rows == columns)
				return std::nullopt;
			return rows ? Layout::rows : Layout::columns;
		}

		/** @brief The domain of the argument that the reduction to @p out reduces (rows: out is its prefix, columns: its suffix). */
		std::optional<Domain> reducedDomain(const ::tvm::ffi::Array<::tvm::Expr>& args, const Domain& out, Layout layout) {
			for (const auto& arg : args) {
				const auto d = domainOf(arg);
				if (!d || d->size() <= out.size())
					continue;
				const size_t offset = layout == Layout::rows ? 0 : d->size() - out.size();
				if (sameDomain(Domain(d->begin() + offset, d->begin() + offset + out.size()), out))
					return d;
			}
			return std::nullopt;
		}
	} // namespace

	bool sameDomain(const Domain& a, const Domain& b) {
		if (a.size() != b.size())
			return false;
		::tvm::sym::Analyzer analyzer;
		for (size_t i = 0; i < a.size(); ++i)
			if (!analyzer->CanProveEqual(a[i], b[i]))
				return false;
		return true;
	}

	Signature signatureOf(const ::tvm::IRModule& mod, const ::tvm::Var& var, const ::tvm::Expr& value) {
		static const auto& callTir = ::tvm::Op::Get("relax.call_tir");
		const auto* call = value.as<::tvm::CallNode>();
		if (call == nullptr || !call->op.same_as(callTir) || call->args.size() != 2)
			return {};
		const auto* gv = call->args[0].as<::tvm::GlobalVarNode>();
		const auto* args = call->args[1].as<::tvm::relax::TupleNode>();
		const auto out = domainOf(var);
		if (gv == nullptr || args == nullptr || !out)
			return {};
		const auto* prim = mod->Lookup(::tvm::ffi::GetRef<::tvm::GlobalVar>(gv)).as<::tvm::tirx::PrimFuncNode>();
		if (prim == nullptr || prim->params.size() != args->fields.size() + 1)
			return {}; // (several results, scalar parameters)
		const auto func = ::tvm::ffi::GetRef<::tvm::tirx::PrimFunc>(prim);
		if (hasAttr(func, "op_pattern", 8) || hasAttr(func, "tirx.is_scheduled", 1))
			return {};
		try {
			auto sch = dl::schedule_of(func);
			const auto infos = dl::try_inline_contiguous_spatial(sch, dl::analysis::normalize_prim_func(sch));
			if (!infos || infos->empty())
				return {};
			bool reduces = false;
			for (const auto& info : *infos)
				reduces |= info.is_reduction();
			if (!reduces)
				return {Kind::pointwise, Layout::rows, *out, *out, *out, ::tvm::relax::HasReshapePattern(func)};
			const auto layout = reductionLayout(sch, *infos);
			if (!layout)
				return {Kind::anchor, Layout::rows, *out, *out, *out};
			if (auto full = reducedDomain(args->fields, *out, *layout))
				return {Kind::reduction, *layout, *out, *out, *full};
			// Rows with an epilogue (e.g. a normalization): the result has the reduced input's domain; the rows are its
			// leading dimensions, as many as the reduction keeps.
			size_t kept = 0;
			for (const auto& info : *infos)
				if (info.is_reduction()) {
					for (const auto& it : info.iters)
						kept += it.kind == 'S' && !::tvm::prim::is_one(it.dom);
					break;
				}
			if (*layout == Layout::rows && kept < out->size())
				return {Kind::reduction, Layout::rows, *out, Domain(out->begin(), out->begin() + kept), *out};
			return {};
		} catch (const std::exception&) {
			return {};
		}
	}

} // namespace mdl::backend::tvm::fusion
