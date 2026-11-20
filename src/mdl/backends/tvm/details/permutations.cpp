#include "./permutations.hpp"

#include "./tvm_compat.hpp"

#include <tvm/relax/expr.h>
#include <tvm/tirx/op.h>

#include <numeric>
#include <utility>

namespace mdl::backend::tvm {

	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	using ::tvm::relax::Expr;
	using ::tvm::relax::ShapeExpr;

	Perm permOf(const Expr& e) {
		Perm perm;
		if (const auto* shape = e.as<::tvm::relax::ShapeExprNode>())
			for (const auto& v : shape->values)
				perm.push_back(intValue(v.as<::tvm::IntImmNode>()));
		return perm;
	}

	Expr permExpr(const Perm& perm) {
		Array<PrimExpr> values;
		for (int64_t p : perm)
			values.push_back(::tvm::IntImm(::tvm::PrimType::Int(64), p));
		return ShapeExpr(values);
	}

	/** @brief permute(xs, perm): entry i is xs[perm[i]] (like relax.permute_dims). */
	Array<PrimExpr> permuted(const Array<PrimExpr>& xs, const Perm& perm) {
		if (perm.empty())
			return xs;
		Array<PrimExpr> out;
		for (int64_t p : perm)
			out.push_back(xs[p]);
		return out;
	}

	/** @brief permute(permute(x, inner), outer) = permute(x, compose(outer, inner)). */
	Perm compose(const Perm& outer, const Perm& inner) {
		if (outer.empty())
			return inner;
		if (inner.empty())
			return outer;
		Perm r;
		for (int64_t p : outer)
			r.push_back(inner[p]);
		return r;
	}

	Perm swapLastTwo(size_t rank) {
		Perm p(rank);
		std::iota(p.begin(), p.end(), 0);
		std::swap(p[rank - 1], p[rank - 2]);
		return p;
	}

	bool isIdentity(const Perm& p) {
		for (size_t i = 0; i < p.size(); ++i)
			if (p[i] != static_cast<int64_t>(i))
				return false;
		return true;
	}

	Perm inverse(const Perm& perm) {
		Perm r(perm.size());
		for (size_t i = 0; i < perm.size(); ++i)
			r[perm[i]] = static_cast<int64_t>(i);
		return r;
	}

	Array<PrimExpr> contiguousStrides(const Array<PrimExpr>& shape) {
		std::vector<PrimExpr> strides(shape.size());
		PrimExpr s = ::tvm::IntImm(::tvm::PrimType::Int(64), 1);
		for (size_t i = shape.size(); i-- > 0;) {
			strides[i] = s;
			s = s * ::tvm::prim::cast(::tvm::PrimType::Int(64), shape[i]);
		}
		return Array<PrimExpr>(strides.begin(), strides.end());
	}

} // namespace mdl::backend::tvm
