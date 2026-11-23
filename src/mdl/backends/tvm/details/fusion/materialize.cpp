#include "./materialize.hpp"

// TVM internals (src/; pinned TVM version): the grouping of FuseOps and FuseOpsByPattern, for any partition.
#include <relax/transform/utils.h>

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace mdl::backend::tvm::fusion {

	namespace {
		namespace relax = ::tvm::relax;
		using Group = relax::GraphPartitioner::Group;

		/** @brief The variables bound in a function (each its own group unless planned otherwise). */
		struct BoundVars : relax::ExprVisitor {
			std::vector<const ::tvm::ffi::Object*> vars;
			using relax::ExprVisitor::VisitBinding_;
			void VisitBinding_(const relax::VarBindingNode* binding) final {
				vars.push_back(binding->var.get());
				relax::ExprVisitor::VisitBinding_(binding);
			}
		};

		/** @brief @p f with each result that @p f also uses itself computed again at its end. */
		relax::Function withTrailingResults(const relax::Function& f) {
			const auto* seq = f->body.as<relax::SeqExprNode>();
			if (seq == nullptr || seq->blocks.empty())
				return f;
			std::unordered_map<const void*, size_t> uses;
			std::unordered_map<const void*, ::tvm::Expr> valueOf;
			for (const auto& block : seq->blocks)
				for (const auto& b : block->bindings)
					if (const auto* vb = b.as<relax::VarBindingNode>()) {
						valueOf.emplace(vb->var.get(), vb->value);
						for (const auto& v : relax::FreeVars(vb->value))
							++uses[v.get()];
					}
			::tvm::ffi::Array<relax::Binding> trailing;
			auto again = [&](const ::tvm::Expr& e) -> ::tvm::Expr {
				const auto* var = e.as<::tvm::VarNode>();
				if (var == nullptr || !uses.contains(var) || !valueOf.contains(var))
					return e;
				const ::tvm::Var result(std::string(var->name), relax::GetType(e));
				trailing.push_back(relax::VarBinding(result, valueOf.at(var)));
				return result;
			};
			::tvm::Expr body = seq->body;
			if (const auto* tuple = body.as<relax::TupleNode>()) {
				::tvm::ffi::Array<::tvm::Expr> fields;
				for (const auto& field : tuple->fields)
					fields.push_back(again(field));
				body = relax::Tuple(fields);
			} else {
				body = again(body);
			}
			if (trailing.empty())
				return f;
			auto blocks = seq->blocks;
			const auto& last = blocks.back();
			auto bindings = last->bindings;
			for (const auto& b : trailing)
				bindings.push_back(b);
			blocks.Set(blocks.size() - 1, last.as<relax::DataflowBlockNode>() ? relax::BindingBlock(relax::DataflowBlock(bindings))
																			  : relax::BindingBlock(bindings));
			return relax::Function(f->params, relax::SeqExpr(blocks, body), f->ret_ty, f->is_pure, f->attrs);
		}
	} // namespace

	::tvm::IRModule materialize(::tvm::IRModule mod, const Partition& groups) {
		std::deque<Group> storage; // (stable addresses)
		std::unordered_map<const ::tvm::ffi::Object*, Group*> partition;
		std::unordered_set<const void*> before;
		for (const auto& [gv, f] : mod->functions) {
			before.insert(gv.get());
			if (const auto* fn = f.as<relax::FunctionNode>()) {
				BoundVars bound;
				bound.VisitExpr(::tvm::ffi::GetRef<relax::Function>(fn));
				for (const auto* v : bound.vars)
					partition[v] = &storage.emplace_back();
			}
		}
		for (const auto& group : groups) {
			Group* g = &storage.emplace_back();
			g->num_nodes = static_cast<uint32_t>(group.size());
			for (const auto& v : group)
				partition[v.get()] = g;
		}
		mod = relax::MakeGroupedFunctions(mod, partition, /*lift_constants=*/true);
		std::vector<std::pair<::tvm::GlobalVar, relax::Function>> updates;
		for (const auto& [gv, f] : mod->functions) {
			const auto* fn = f.as<relax::FunctionNode>();
			if (fn == nullptr)
				continue;
			const auto function = ::tvm::ffi::GetRef<relax::Function>(fn);
			updates.emplace_back(
					gv, before.contains(gv.get()) ? relax::EliminateCommonSubexpr(function).as_or_throw<relax::Function>()
												  : withTrailingResults(function)
			);
		}
		for (const auto& [gv, f] : updates)
			mod->Update(gv, f);
		return mod;
	}

} // namespace mdl::backend::tvm::fusion
