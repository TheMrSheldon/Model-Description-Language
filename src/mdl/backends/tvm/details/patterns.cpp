#include "./legalize.hpp"
#include "./pattern_registry.hpp"

#include <tvm/arith/analyzer.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/struct_info.h>

#include <cmath>
#include <optional>

namespace mdl::backend::tvm {

	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	using ::tvm::ffi::Map;
	using namespace ::tvm::relax;

	namespace {
		using Bindings = Map<Var, Expr>;

		/** @brief Looks through variables to the expressions they are bound to. */
		Expr resolve(Expr e, const Bindings& bindings) {
			while (const auto* var = e.as<VarNode>()) {
				auto it = bindings.find(::tvm::ffi::GetRef<Var>(var));
				if (it == bindings.end())
					break;
				e = (*it).second;
			}
			return e;
		}

		/** @brief The value of a scalar constant (0-d / single-element float constant or PrimValue). */
		std::optional<double> scalarConstant(const Expr& e) {
			if (const auto* c = e.as<ConstantNode>()) {
				const auto& data = c->data;
				if (data.Shape().Product() != 1 || data->dtype.code != kDLFloat)
					return std::nullopt;
				if (data->dtype.bits == 32)
					return static_cast<const float*>(data->data)[0];
				if (data->dtype.bits == 64)
					return static_cast<const double*>(data->data)[0];
				return std::nullopt;
			}
			if (const auto* pv = e.as<PrimValueNode>()) {
				if (const auto* f = pv->value.as<::tvm::FloatImmNode>())
					return f->value;
				if (const auto* i = pv->value.as<::tvm::IntImmNode>())
					return static_cast<double>(i->value);
			}
			return std::nullopt;
		}

		bool near(std::optional<double> value, double expected) {
			return value && std::abs(*value - expected) <= 1e-6 * std::max(1.0, std::abs(expected));
		}

		std::optional<Array<PrimExpr>> shapeOf(const Expr& e) {
			const auto* info = GetStructInfoAs<TensorStructInfoNode>(e);
			if (info == nullptr || !info->GetShape().defined())
				return std::nullopt;
			return info->GetShape().value();
		}

		size_t rankOf(const Expr& e) {
			auto shape = shapeOf(e);
			return shape ? shape->size() : 0;
		}

		/** @brief Whether @p call reduces over (only) the last axis with the given keepdims. */
		bool reducesLastAxis(const Expr& call, bool keepdims) {
			const auto* c = call.as<CallNode>();
			const auto* attrs = c ? c->attrs.as<StatisticalAttrs>() : nullptr;
			if (attrs == nullptr || attrs->keepdims != keepdims || !attrs->axis.defined() ||
				attrs->axis.value().size() != 1)
				return false;
			int64_t axis = attrs->axis.value()[0]->value;
			return axis == -1 || axis == static_cast<int64_t>(rankOf(c->args[0])) - 1;
		}

		/** @brief Whether @p call is a permute_dims that swaps the last two axes (and keeps all others). */
		bool swapsLastTwoAxes(const Expr& call) {
			const auto* c = call.as<CallNode>();
			const auto* attrs = c ? c->attrs.as<PermuteDimsAttrs>() : nullptr;
			if (attrs == nullptr)
				return false;
			const size_t rank = rankOf(c->args[0]);
			if (rank < 2)
				return false;
			if (!attrs->axes.defined()) // reverses all axes
				return rank == 2;
			const auto& axes = attrs->axes.value();
			if (axes.size() != rank)
				return false;
			for (size_t i = 0; i < rank; ++i) {
				size_t expected = (i + 2 < rank) ? i : (i + 2 == rank ? rank - 1 : rank - 2);
				int64_t axis = axes[i]->value;
				if (axis < 0)
					axis += static_cast<int64_t>(rank);
				if (axis != static_cast<int64_t>(expected))
					return false;
			}
			return true;
		}

		/** @brief Whether the last dimension of @p e equals @p n. */
		bool lastDimEquals(const Expr& e, double n) {
			auto shape = shapeOf(e);
			if (!shape || shape->empty())
				return false;
			const auto* imm = shape->back().as<::tvm::IntImmNode>();
			return imm && near(n, static_cast<double>(imm->value));
		}

		/** @brief Whether @p e is a 1-D tensor whose length equals the last dimension of @p of. */
		bool isVectorOverLastDim(const Expr& e, const Expr& of) {
			auto shape = shapeOf(e), ofShape = shapeOf(of);
			if (!shape || !ofShape || shape->size() != 1 || ofShape->empty())
				return false;
			return ::tvm::arith::Analyzer().CanProveEqual((*shape)[0], ofShape->back());
		}

		Expr callOp(const char* name, Array<Expr> args, ::tvm::Attrs attrs = {}) {
			return Call(::tvm::Op::Get(name), args, attrs);
		}

		// ── softmax: exp(x - max(x)) / sum(exp(x - max(x))) ────────────────────────────────────────────────

		PatternEntry makeSoftmax() {
			auto x = Wildcard();
			auto maxX = IsOp("relax.max")(x);
			auto e = IsOp("relax.exp")(IsOp("relax.subtract")(x, maxX));
			auto sumE = IsOp("relax.sum")(e);
			DFPattern pattern = IsOp("relax.divide")(e, sumE);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				if (!reducesLastAxis(at(maxX), true) || !reducesLastAxis(at(sumE), true))
					return call;
				static const auto softmax = ::tvm::ffi::Function::GetGlobalRequired("relax.op.nn.softmax");
				return softmax(m.at(x), -1).cast<Expr>();
			};
			return {"softmax", pattern, rewriter};
		}

		// ── GELU: x * 0.5 * (1 + erf(x / sqrt(2))) ────────────────────────────────────────────────────────

		PatternEntry makeGelu() {
			auto x = Wildcard(), half = Wildcard(), one = Wildcard(), sqrt2 = Wildcard();
			auto erfTerm = IsOp("relax.erf")(IsOp("relax.divide")(x, sqrt2));
			DFPattern pattern =
					IsOp("relax.multiply")(IsOp("relax.multiply")(x, half), IsOp("relax.add")(one, erfTerm));
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				if (!near(scalarConstant(at(half)), 0.5) || !near(scalarConstant(at(one)), 1.0) ||
					!near(scalarConstant(at(sqrt2)), std::sqrt(2.0)))
					return call;
				static const auto gelu = ::tvm::ffi::Function::GetGlobalRequired("relax.op.nn.gelu");
				return gelu(m.at(x)).cast<Expr>();
			};
			return {"gelu", pattern, rewriter};
		}

		// ── layer norm (std.LayerNorm) ──────────────────────────────────────────────────────────────────
		//   mu = reshape(sum(x) / n)          xc = x - mu
		//   s  = rsqrt(reshape(sum(xc * xc) / n) + eps)
		//   out = xc * s * gamma + beta

		PatternEntry makeLayerNorm() {
			auto x = Wildcard(), gamma = Wildcard(), beta = Wildcard();
			auto n1 = Wildcard(), n2 = Wildcard(), eps = Wildcard();
			auto sumX = IsOp("relax.sum")(x);
			auto mu = IsOp("relax.reshape")(IsOp("relax.divide")(sumX, n1), Wildcard());
			auto xc = IsOp("relax.subtract")(x, mu);
			auto sumSq = IsOp("relax.sum")(IsOp("relax.multiply")(xc, xc));
			auto var = IsOp("relax.reshape")(IsOp("relax.divide")(sumSq, n2), Wildcard());
			auto s = IsOp("relax.rsqrt")(IsOp("relax.add")(var, eps));
			DFPattern pattern =
					IsOp("relax.add")(IsOp("relax.multiply")(IsOp("relax.multiply")(xc, s), gamma), beta);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				const Expr& in = m.at(x);
				auto n = scalarConstant(at(n1));
				auto epsilon = scalarConstant(at(eps));
				if (!reducesLastAxis(at(sumX), false) || !reducesLastAxis(at(sumSq), false) || !n ||
					!lastDimEquals(in, *n) || !near(scalarConstant(at(n2)), *n) || !epsilon ||
					!isVectorOverLastDim(m.at(gamma), in) || !isVectorOverLastDim(m.at(beta), in))
					return call;
				static const auto layerNorm = ::tvm::ffi::Function::GetGlobalRequired("relax.op.nn.layer_norm");
				return layerNorm(in, m.at(gamma), m.at(beta), Array<::tvm::Integer>{-1}, *epsilon, true, true)
						.cast<Expr>();
			};
			return {"layer_norm", pattern, rewriter};
		}

		// ── matmul variants → mdl.matmul ───────────────────────────────────────────────────────────────

		/** a @ W^T + b and a @ W + b with a 2-D weight W and a bias over the output features. */
		PatternEntry makeLinear(bool transposed) {
			auto a = Wildcard(), w = Wildcard(), bias = Wildcard();
			auto perm = IsOp("relax.permute_dims")(w);
			DFPattern weight = transposed ? DFPattern(perm) : DFPattern(w);
			auto product = IsOp("relax.matmul")(a, weight);
			DFPattern pattern = IsOp("relax.add")(product, bias);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				if (rankOf(m.at(w)) != 2 || rankOf(m.at(a)) < 2 || (transposed && !swapsLastTwoAxes(at(perm))) ||
					!isVectorOverLastDim(m.at(bias), m.at(product)))
					return call;
				return makeMatmul(m.at(a), m.at(w), m.at(bias), transposed, 1.0);
			};
			return {transposed ? "linear (W^T)" : "linear", pattern, rewriter};
		}

		/** a @ B^T (any rank, e.g. attention scores q @ k^T). */
		PatternEntry makeMatmulTransposed() {
			auto a = Wildcard(), b = Wildcard();
			auto perm = IsOp("relax.permute_dims")(b);
			DFPattern pattern = IsOp("relax.matmul")(a, perm);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				if (rankOf(m.at(a)) < 2 || rankOf(m.at(b)) < 2 || !swapsLastTwoAxes(at(perm)))
					return call;
				return makeMatmul(m.at(a), m.at(b), std::nullopt, true, 1.0);
			};
			return {"matmul (B^T)", pattern, rewriter};
		}

		/** (a @ op(B)) / c → mdl.matmul with alpha = 1/c (e.g. the attention scale 1/sqrt(d)). */
		PatternEntry makeScaledMatmul() {
			auto product = IsOp("mdl.matmul")(Wildcard(), Wildcard(), Wildcard(), Wildcard(), Wildcard());
			auto c = Wildcard();
			DFPattern pattern = IsOp("relax.divide")(product, c);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				auto divisor = scalarConstant(at(c));
				auto productCall = at(product);
				const auto* mm = productCall.as<CallNode>();
				auto alpha = scalarConstant(mm->args[4]);
				// Only without bias: (a @ b + bias) / c != alpha * (a @ b) + bias.
				if (!divisor || *divisor == 0.0 || !alpha || mm->args[2].as<PrimValueNode>() == nullptr)
					return call;
				const auto* transB = mm->args[3].as<PrimValueNode>()->value.as<::tvm::IntImmNode>();
				return makeMatmul(mm->args[0], mm->args[1], std::nullopt, transB->value != 0, *alpha / *divisor);
			};
			return {"matmul scale", pattern, rewriter};
		}
	} // namespace

	std::vector<PatternEntry> defaultPatterns() {
		return {
				makeSoftmax(),			// before the matmul patterns: the attention's softmax input is a divide
				makeGelu(),
				makeLayerNorm(),
				makeLinear(true),		// a @ W^T + b (std.Linear)
				makeLinear(false),		// a @ W + b
				makeMatmulTransposed(), // remaining a @ B^T
				makeScaledMatmul(),		// (a @ B^T) / c
		};
	}

} // namespace mdl::backend::tvm
