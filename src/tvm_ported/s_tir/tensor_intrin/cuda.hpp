// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/tensor_intrin/cuda.py
// (conventions: src/tvm_ported/README.md)
#pragma once

/**
 * @file cuda.hpp
 * @brief Intrinsics for tensorization on NVIDIA GPU.
 * @details Ported: the WMMA intrinsics ("WMMA intrinsics" section of cuda.py) and get_wmma_intrin_group. Not ported:
 * the ldmatrix/MMA intrinsics (get_ldmatrix_intrin, get_mma_intrin, ..., get_mma_intrin_group).
 */

#include "../../tvm.hpp"

#include <map>
#include <string>

namespace mdl::tvm_ported::tensor_intrin::cuda {

	/** @brief A tensor intrinsic: its description and implementation (Python's `(desc, impl)` tuples). */
	struct Intrin {
		PrimFunc desc, impl;
	};

	Intrin get_wmma_load_intrin(
			int m_dim, int n_dim, int k_dim, const std::string& dtype, const std::string& shared_scope, bool is_b,
			bool is_col_major
	);
	Intrin get_wmma_fill_intrin(int m_dim, int n_dim, int k_dim, const std::string& dtype);
	Intrin get_wmma_store_intrin(int m_dim, int n_dim, int k_dim, const std::string& dtype, const std::string& scope);
	Intrin get_wmma_sync_intrin(
			int m_dim, int n_dim, int k_dim, const std::string& in_dtype, const std::string& out_dtype,
			bool b_transposed
	);

	/**
	 * @brief Registers the WMMA intrinsics with TVM (TensorIntrin.register; in Python, at import of the module).
	 * Idempotent.
	 */
	void register_wmma_intrinsics();

	/**
	 * @brief Get a group of intrinsics for wmma tensor core with the given configurations.
	 * @return the intrinsic names by role: init, load_a, load_b, compute, store.
	 */
	std::map<std::string, std::string> get_wmma_intrin_group(
			const std::string& load_scope, const std::string& store_scope, const std::string& in_dtype,
			const std::string& out_dtype, bool trans_b
	);

} // namespace mdl::tvm_ported::tensor_intrin::cuda
