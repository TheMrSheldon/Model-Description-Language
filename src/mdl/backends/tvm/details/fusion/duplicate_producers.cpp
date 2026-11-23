#include "./duplicate_producers.hpp"

#include "./planner.hpp"
#include "./signature.hpp"

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>
#include <tvm/tirx/function.h>

#include <unordered_map>

namespace mdl::backend::tvm::fusion {

	namespace {
		namespace relax = ::tvm::relax;
		using Kind = Signature::Kind;

		constexpr size_t kMaxExprNodes = 64; ///< of the kernel's computation, for a copy to be cheap
		constexpr int kMaxChain = 8;		 ///< shared producers copied along a chain

		bool cheap(const ::tvm::IRModule& mod, const ::tvm::Expr& value) {
			const auto* call = value.as<::tvm::CallNode>();
			const auto* gv = call ? call->args[0].as<::tvm::GlobalVarNode>() : nullptr;
			const auto* prim = gv ? mod->Lookup(::tvm::ffi::GetRef<::tvm::GlobalVar>(gv)).as<::tvm::tirx::PrimFuncNode>() : nullptr;
			if (prim == nullptr)
				return false;
			size_t nodes = 0;
			::tvm::ffi::StructuralWalk<::tvm::ffi::WalkOrder::kPreOrder>(
					prim->body, [&](const ::tvm::PrimExpr&) -> ::tvm::ffi::Expected<::tvm::ffi::WalkResult> {
						++nodes;
						return ::tvm::ffi::WalkResult::Advance();
					}
			);
			return nodes <= kMaxExprNodes;
		}

		relax::BindingBlock duplicate(const ::tvm::IRModule& mod, const relax::DataflowBlock& block, const std::unordered_set<const void*>& usedAfter) {
			const auto& bindings = block->bindings;
			std::unordered_map<const void*, size_t> index;
			std::vector<Signature> signatures;
			std::vector<std::vector<size_t>> readers(bindings.size());
			for (size_t i = 0; i < bindings.size(); ++i) {
				const auto* vb = bindings[i].as<relax::VarBindingNode>();
				signatures.push_back(vb ? signatureOf(mod, vb->var, vb->value) : Signature{});
				if (vb != nullptr)
					for (const auto& v : relax::FreeVars(vb->value))
						if (auto it = index.find(v.get()); it != index.end())
							readers[it->second].push_back(i);
				index.emplace(bindings[i]->var.get(), i);
			}
			// The producers to copy, and for each reader the copy it reads.
			std::unordered_map<size_t, std::vector<size_t>> copiesFor; // reader -> producers
			for (size_t p = 0; p < bindings.size(); ++p) {
				const Signature& s = signatures[p];
				if (s.kind != Kind::pointwise || readers[p].size() < 2 || usedAfter.contains(bindings[p]->var.get()) ||
					!bindings[p]->var.as<relax::DataflowVarNode>() || !cheap(mod, bindings[p].as_or_throw<relax::VarBinding>()->value))
					continue;
				bool fusible = true;
				for (size_t r : readers[p]) {
					const Signature& t = signatures[r];
					fusible &= (t.kind == Kind::pointwise && sameDomain(t.out, s.out)) ||
							   (t.kind == Kind::reduction && sameDomain(t.full, s.out));
				}
				if (fusible)
					for (size_t r : readers[p])
						copiesFor[r].push_back(p);
			}
			if (copiesFor.empty())
				return block;
			::tvm::ffi::Array<relax::Binding> result;
			for (size_t i = 0; i < bindings.size(); ++i) {
				auto it = copiesFor.find(i);
				if (it == copiesFor.end()) {
					result.push_back(bindings[i]);
					continue;
				}
				::tvm::ffi::Map<::tvm::Var, ::tvm::Expr> rename;
				for (size_t p : it->second) {
					const auto& producer = bindings[p].as_or_throw<relax::VarBinding>();
					const relax::DataflowVar copy(std::string(producer->var->name), relax::GetType(producer->var));
					result.push_back(relax::VarBinding(copy, producer->value));
					rename.Set(producer->var, copy);
				}
				const auto& reader = bindings[i].as_or_throw<relax::VarBinding>();
				result.push_back(relax::VarBinding(reader->var, relax::Bind(reader->value, rename)));
			}
			return relax::DataflowBlock(result);
		}
	} // namespace

	::tvm::transform::Pass duplicateProducers() {
		return relax::transform::CreateFunctionPass(
				[](relax::Function f, ::tvm::IRModule mod, ::tvm::transform::PassContext) {
					const auto* seq = f->body.as<relax::SeqExprNode>();
					if (seq == nullptr || f->HasNonzeroAttr(relax::attr::kPrimitive))
						return f;
					::tvm::ffi::Array<relax::BindingBlock> blocks;
					for (size_t k = 0; k < seq->blocks.size(); ++k) {
						const auto* dataflow = seq->blocks[k].as<relax::DataflowBlockNode>();
						if (dataflow == nullptr) {
							blocks.push_back(seq->blocks[k]);
							continue;
						}
						// (until no copy is made: copies of a shared chain's last value share its producers)
						relax::BindingBlock block = ::tvm::ffi::GetRef<relax::DataflowBlock>(dataflow);
						for (int depth = 0; depth < kMaxChain; ++depth) {
							auto copied = duplicate(mod, block.as_or_throw<relax::DataflowBlock>(), usedAfter(*seq, k));
							if (copied.same_as(block))
								break;
							block = copied;
						}
						blocks.push_back(block);
					}
					return relax::Function(f->params, relax::SeqExpr(blocks, seq->body), f->ret_ty, f->is_pure, f->attrs);
				},
				0, "mdl.DuplicateProducers", {}
		);
	}

} // namespace mdl::backend::tvm::fusion
