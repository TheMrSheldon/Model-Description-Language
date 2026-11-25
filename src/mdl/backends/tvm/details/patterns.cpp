#include "./legalize.hpp"
#include "./pattern_registry.hpp"
#include "./tvm_compat.hpp"
#include <cstring>

#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/type.h>
#include <tvm/sym/analyzer.h>

#include <cmath>
#include <optional>

namespace mdl::backend::tvm {

	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	using ::tvm::ffi::Map;
	using namespace ::tvm::relax;
	using ::tvm::Call;
	using ::tvm::CallNode;
	using ::tvm::Expr;
	using ::tvm::Type;
	using ::tvm::Var;
	using ::tvm::VarNode;

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

		bool near(std::optional<double> value, double expected) {
			return value && std::abs(*value - expected) <= 1e-6 * std::max(1.0, std::abs(expected));
		}

		/** @brief Whether @p e is a scalar constant equal to @p expected within the precision of its type. */
		bool near(const Expr& e, double expected) {
			auto value = scalarConstant(e);
			const auto c = tensorConstant(e);
			const bool half = c && (*c)->dtype.bits == 16; // float16: 11, bfloat16: 8 significant bits
			const double tolerance = half ? ((*c)->dtype.code == kDLBfloat ? 4e-3 : 5e-4) : 1e-6;
			return value && std::abs(*value - expected) <= tolerance * std::max(1.0, std::abs(expected));
		}

		size_t rankOf(const Expr& e) {
			auto shape = knownShape(e);
			return shape ? shape->size() : 0;
		}

		/** @brief Whether @p call reduces over (only) the last axis with the given keepdims. */
		bool reducesLastAxis(const Expr& call, bool keepdims) {
			const auto* c = call.as<CallNode>();
			const auto* attrs = c ? c->attrs.as<StatisticalAttrs>() : nullptr;
			if (attrs == nullptr || attrs->keepdims != keepdims || !attrs->axis.defined() ||
				attrs->axis.value().size() != 1)
				return false;
			int64_t axis = attrs->axis.value()[0];
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
				int64_t axis = axes[i];
				if (axis < 0)
					axis += static_cast<int64_t>(rank);
				if (axis != static_cast<int64_t>(expected))
					return false;
			}
			return true;
		}

		/** @brief Whether the last dimension of @p e equals @p n. */
		bool lastDimEquals(const Expr& e, double n) {
			auto shape = knownShape(e);
			if (!shape || shape->empty())
				return false;
			const auto* imm = shape->back().as<::tvm::IntImmNode>();
			return imm && near(n, static_cast<double>(imm->value));
		}

		/** @brief Whether @p e is a 1-D tensor whose length equals the last dimension of @p of. */
		bool isVectorOverLastDim(const Expr& e, const Expr& of) {
			auto shape = knownShape(e), ofShape = knownShape(of);
			if (!shape || !ofShape || shape->size() != 1 || ofShape->empty())
				return false;
			return ::tvm::sym::Analyzer()->CanProveEqual((*shape)[0], ofShape->back());
		}

		Expr callOp(const char* name, Array<Expr> args, ::tvm::Attrs attrs = {}) {
			return Call(Type::Missing(), ::tvm::Op::Get(name), args, attrs);
		}

		// ── softmax: exp(x - max(x)) / sum(exp(x - max(x))) ────────────────────────────────────────────────

		PatternEntry makeSoftmax() {
			// The reductions keep the last axis: with keepdims, or (as MDL emits them) without and restored by a
			// reshape to [..., 1].
			auto x = Wildcard();
			auto maxX = IsOp("relax.max")(x);
			auto maxKept = IsOp("relax.reshape")(maxX, Wildcard());
			auto e = IsOp("relax.exp")(IsOp("relax.subtract")(x, maxX | maxKept));
			auto sumE = IsOp("relax.sum")(e);
			auto sumKept = IsOp("relax.reshape")(sumE, Wildcard());
			DFPattern pattern = IsOp("relax.divide")(e, sumE | sumKept);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				auto kept = [&](const DFPattern& reduction, const DFPattern& reshape) {
					if (!m.count(reshape))
						return reducesLastAxis(at(reduction), true);
					return reducesLastAxis(at(reduction), false) && rankOf(m.at(reshape)) == rankOf(m.at(x)) &&
						   lastDimEquals(m.at(reshape), 1);
				};
				if (!kept(maxX, maxKept) || !kept(sumE, sumKept))
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
				if (!near(at(half), 0.5) || !near(at(one), 1.0) || !near(at(sqrt2), std::sqrt(2.0)))
					return call;
				static const auto gelu = ::tvm::ffi::Function::GetGlobalRequired("relax.op.nn.gelu");
				return gelu(m.at(x)).cast<Expr>();
			};
			return {"gelu", pattern, rewriter};
		}

		// ── GELU's gradient (std.gelu_gradient): dy * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-0.5 * x * x) / sqrt(2π))

		PatternEntry makeGeluGradient() {
			auto x = Wildcard(), dy = Wildcard();
			auto sqrt2 = Wildcard(), one = Wildcard(), half = Wildcard(), minusHalf = Wildcard(), density = Wildcard();
			auto cdf = IsOp("relax.multiply")(
					half, IsOp("relax.add")(one, IsOp("relax.erf")(IsOp("relax.divide")(x, sqrt2)))
			);
			auto gauss = IsOp("relax.exp")(IsOp("relax.multiply")(IsOp("relax.multiply")(minusHalf, x), x));
			auto pdf = IsOp("relax.multiply")(IsOp("relax.multiply")(x, gauss), density);
			DFPattern pattern = IsOp("relax.multiply")(dy, IsOp("relax.add")(cdf, pdf));
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				if (!near(at(sqrt2), std::sqrt(2.0)) || !near(at(one), 1.0) || !near(at(half), 0.5) ||
					!near(at(minusHalf), -0.5) || !near(at(density), 1.0 / std::sqrt(2.0 * M_PI)))
					return call;
				return ::mdl::backend::tvm::makeGeluBackward(m.at(x), m.at(dy));
			};
			return {"gelu backward", pattern, rewriter};
		}

		// ── layer norm (std.LayerNorm) ──────────────────────────────────────────────────────────────────
		//   mu = reshape(sum(x) / n)          xc = x - mu
		//   s  = rsqrt(reshape(sum(xc * xc) / n) + eps)
		//   out = xc * s * gamma + beta

		PatternEntry makeLayerNorm() {
			auto x = Wildcard(), gamma = Wildcard(), beta = Wildcard();
			auto n1 = Wildcard(), n2 = Wildcard(), eps = Wildcard();
			auto sumX = IsOp("relax.sum")(x);
			auto meanX = IsOp("relax.divide")(sumX, n1);
			auto mu = IsOp("relax.reshape")(meanX, Wildcard());
			auto xc = IsOp("relax.subtract")(x, mu);
			auto sq = IsOp("relax.multiply")(xc, xc);
			auto sumSq = IsOp("relax.sum")(sq);
			auto meanSq = IsOp("relax.divide")(sumSq, n2);
			auto var = IsOp("relax.reshape")(meanSq, Wildcard());
			auto varEps = IsOp("relax.add")(var, eps);
			auto s = IsOp("relax.rsqrt")(varEps);
			auto xhat = IsOp("relax.multiply")(xc, s);
			auto scaled = IsOp("relax.multiply")(xhat, gamma);
			DFPattern pattern = IsOp("relax.add")(scaled, beta);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				const Expr& in = m.at(x);
				auto n = scalarConstant(at(n1));
				auto epsilon = scalarConstant(at(eps));
				if (!reducesLastAxis(at(sumX), false) || !reducesLastAxis(at(sumSq), false) || !n ||
					!lastDimEquals(in, *n) || !near(at(n2), *n) || !epsilon || !isVectorOverLastDim(m.at(gamma), in) ||
					!isVectorOverLastDim(m.at(beta), in))
					return call;
				static const auto layerNorm = ::tvm::ffi::Function::GetGlobalRequired("relax.op.nn.layer_norm");
				return layerNorm(in, m.at(gamma), m.at(beta), Array<int64_t>{-1}, *epsilon, true, true).cast<Expr>();
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

		/** A^T @ b (e.g. weight gradients x^T @ g): a 2-D a with a 2-D b, or batched a and b of equal rank. */
		PatternEntry makeMatmulTransposedA() {
			auto a = Wildcard(), b = Wildcard();
			auto perm = IsOp("relax.permute_dims")(a);
			DFPattern pattern = IsOp("relax.matmul")(perm, b);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				const int ra = rankOf(m.at(a)), rb = rankOf(m.at(b));
				if (ra < 2 || rb < 2 || !swapsLastTwoAxes(at(perm)) || (rb == 2 && ra != 2) || (rb > 2 && ra != rb))
					return call;
				return makeMatmul(m.at(a), m.at(b), std::nullopt, false, 1.0, /*transA=*/true);
			};
			return {"matmul (A^T)", pattern, rewriter};
		}

		/** The remaining a @ b (any rank ≥ 2, e.g. attention probabilities @ values): permutations fold into it. */
		PatternEntry makePlainMatmul() {
			auto a = Wildcard(), b = Wildcard();
			DFPattern pattern = IsOp("relax.matmul")(a, b);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings&) -> Expr {
				if (rankOf(m.at(a)) < 2 || rankOf(m.at(b)) < 2)
					return call;
				return makeMatmul(m.at(a), m.at(b), std::nullopt, false, 1.0);
			};
			return {"matmul", pattern, rewriter};
		}

		/** (a @ op(B)) / c → mdl.matmul with alpha = 1/c (e.g. the attention scale 1/sqrt(d)). */
		PatternEntry makeScaledMatmul() {
			auto product = IsOp("mdl.matmul")(
					Wildcard(), Wildcard(), Wildcard(), Wildcard(), Wildcard(), Wildcard(), Wildcard(), Wildcard(),
					Wildcard(), Wildcard(), Wildcard()
			);
			auto c = Wildcard();
			DFPattern pattern = IsOp("relax.divide")(product, c);
			RewriterFn rewriter = [=](Expr call, Map<DFPattern, Expr> m, const Bindings& bindings) -> Expr {
				auto at = [&](const DFPattern& p) { return resolve(m.at(p), bindings); };
				auto divisor = scalarConstant(at(c));
				auto productCall = at(product);
				const auto* mm = productCall.as<CallNode>();
				auto alpha = scalarConstant(mm->args[6]);
				// Only without bias: (a @ b + bias) / c != alpha * (a @ b) + bias.
				if (!divisor || *divisor == 0.0 || !alpha || !scalarValue(mm->args[2]))
					return call;
				auto args = mm->args;
				args.Set(6, ::tvm::FloatImm(::tvm::PrimType::Float(64), *alpha / *divisor));
				return Call(Type::Missing(), mm->op, args, mm->attrs, mm->ty_args);
			};
			return {"matmul scale", pattern, rewriter};
		}
	} // namespace

	std::vector<PatternEntry> defaultPatterns() {
		return {
				makeSoftmax(), // before the matmul patterns: the attention's softmax input is a divide
				makeGelu(),
				makeGeluGradient(),
				makeLayerNorm(),
				makeLinear(true),		 // a @ W^T + b (std.Linear)
				makeLinear(false),		 // a @ W + b
				makeMatmulTransposed(),	 // remaining a @ B^T
				makeMatmulTransposedA(), // A^T @ b
				makePlainMatmul(),		 // remaining a @ b
				makeScaledMatmul(),		 // (a @ B^T) / c
		};
	}

} // namespace mdl::backend::tvm
