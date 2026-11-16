#pragma once

/**
 * @file tvm_compat.hpp
 * @brief Small helpers over TVM's unified IR (0.27+): tensor constants are `GenericConst`s holding a runtime tensor,
 * scalars are `PrimExpr`s used directly as Relax expressions, and element types are `PrimType`s.
 */

#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/tensor.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>
#include <stdexcept>

namespace mdl::backend::tvm {

	/** @brief The tensor of a Relax tensor constant, if @p e is one. */
	inline std::optional<::tvm::runtime::Tensor> tensorConstant(const ::tvm::Expr& e) {
		if (const auto* c = e.as<::tvm::GenericConstNode>())
			if (auto tensor = c->value.try_cast<::tvm::runtime::Tensor>())
				return *tensor;
		return std::nullopt;
	}

	/** @brief The value of a scalar (a `PrimExpr` used as a Relax expression), if @p e is one. */
	inline std::optional<::tvm::PrimExpr> scalarValue(const ::tvm::Expr& e) {
		if (e.defined() && e->ty.as<::tvm::PrimTypeNode>() != nullptr)
			return e.as_or_throw<::tvm::PrimExpr>();
		return std::nullopt;
	}

	/** @brief @p shape with the variables of @p vars replaced by their values. */
	inline ::tvm::ffi::Array<::tvm::PrimExpr> substitute(
			const ::tvm::ffi::Array<::tvm::PrimExpr>& shape, const ::tvm::ffi::Map<::tvm::Var, ::tvm::PrimExpr>& vars
	) {
		::tvm::ffi::Array<::tvm::PrimExpr> result;
		for (const auto& d : shape)
			result.push_back(::tvm::tirx::SubstituteWithDataTypeLegalization(d, [&](const ::tvm::Var& v) {
				auto it = vars.find(v);
				return it == vars.end() ? ::tvm::ffi::Optional<::tvm::PrimExpr>() : (*it).second;
			}));
		return result;
	}

	/** @brief The value of an integer constant. */
	inline int64_t intValue(const ::tvm::IntImmNode* imm) { return static_cast<int64_t>(imm->value); }

	/** @brief The value of an integer scalar expression (throws if @p e is none). */
	inline int64_t intValue(const ::tvm::Expr& e) {
		const auto* imm = e.as<::tvm::IntImmNode>();
		if (imm == nullptr)
			throw std::runtime_error("mdl backend: expected an integer scalar");
		return intValue(imm);
	}

	/** @brief The tensor type of @p e, if it is a tensor. */
	inline const ::tvm::relax::TensorTypeNode* tensorInfo(const ::tvm::Expr& e) {
		return ::tvm::relax::GetTypeAs<::tvm::relax::TensorTypeNode>(e);
	}

	/** @brief The shape of the tensor @p e, if it is known. */
	inline std::optional<::tvm::ffi::Array<::tvm::PrimExpr>> knownShape(const ::tvm::Expr& e) {
		const auto* info = tensorInfo(e);
		if (info == nullptr || !info->GetShape().defined())
			return std::nullopt;
		return info->GetShape().value();
	}

	inline bool isFloat(const ::tvm::PrimType& t) { return t.MatchesCode(kDLFloat); }
	inline bool isBFloat16(const ::tvm::PrimType& t) { return t.MatchesElementType(kDLBfloat, 16); }

} // namespace mdl::backend::tvm
