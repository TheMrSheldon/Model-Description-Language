// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/gpu/matmul.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
// A GEMM schedule rule for GPU operators.
// Ported: the helpers, `MatmulTensorization` and `Matmul`. Not ported: `MatmulInt8Tensorization`, `MetalMatmul`, and `Matmul.sch_outer_reduction` (OpenCL on Android/Adreno).
#include "./gpu.hpp"

#include "../analysis/common_analysis.hpp"
#include "../base/utils.hpp"
#include "../../tensor_intrin/cuda.hpp"

#include <tvm/tirx/analysis.h>

#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace mdl::tvm_ported::dlight::gpu {

	namespace {
		Array<SBlockRV> _collect_producers(const Schedule& sch, const SBlockRV& block) {
			Array<SBlockRV> result;
			for (const auto& producer : sch->GetProducers(block)) {
				result.push_back(producer);
				for (const auto& p : _collect_producers(sch, producer))
					result.push_back(p);
			}
			return result;
		}

		Array<SBlockRV> _collect_consumers(const Schedule& sch, const SBlockRV& block) {
			Array<SBlockRV> result;
			for (const auto& consumer : sch->GetConsumers(block)) {
				result.push_back(consumer);
				for (const auto& c : _collect_consumers(sch, consumer))
					result.push_back(c);
			}
			return result;
		}

		void auto_inline_producers(const Schedule& sch, const SBlockRV& block) {
			while (true) {
				int inlined_cnt = 0;
				for (const auto& producer : _collect_producers(sch, block)) {
					try {
						sch->ComputeInline(producer);
						inlined_cnt += 1;
					} catch (const std::exception&) {
						continue;
					}
				}
				if (inlined_cnt == 0)
					return;
			}
		}

		void auto_inline_consumers(const Schedule& sch, const SBlockRV& block) {
			while (true) {
				int inlined_cnt = 0;
				const auto consumers = _collect_consumers(sch, block);
				for (const auto& consumer : consumers) {
					try {
						sch->ComputeInline(consumer);
						inlined_cnt += 1;
					} catch (const std::exception&) {
						continue;
					}
				}
				for (const auto& consumer : consumers) {
					try {
						sch->ReverseComputeInline(consumer);
						inlined_cnt += 1;
					} catch (const std::exception&) {
						continue;
					}
				}
				if (inlined_cnt == 0)
					return;
			}
		}

		void auto_inline_consumer_chain(const Schedule& sch, const SBlockRV& block) {
			auto_inline_consumers(sch, block);
			const auto remaining_consumers = sch->GetConsumers(block);

			if (!remaining_consumers.empty()) {
				for (const auto& c : remaining_consumers) {
					for (const auto& p : sch->GetProducers(c)) {
						if (!sch->Get(p).same_as(sch->Get(block))) {
							auto_inline_producers(sch, p);
							sch->ComputeInline(p);
						}
					}
				}
				auto_inline_consumers(sch, block);
			}

			if (!sch->GetConsumers(block).empty())
				throw std::logic_error(
						"There are some consumers of the cache-write stage that are not properly inlined."
				);
		}

		enum class IterKind {
			/**
			 * Iter kinds for GEMM-liked programs. We can simplify the computation to C[S, I, J] += A[S, I, K] *
			 * B[S, J, K], where `I, J, K` are fundamental axes for gemm and `S` represents all other spatial axes
			 * (e.g. batches) kIter_S: spatial axes; kIter_I: I axes; kIter_J: J axes; kIter_K: K axes; kIter_T: trivial
			 * axes (i.e. with extent 1)
			 */
			kIter_S = 0,
			kIter_I = 1,
			kIter_J = 2,
			kIter_K = 3,
			kIter_T = 4,
		};

		struct IterTrait {
			IterKind kind;
			PrimExpr extent;
		};

		bool _is_one(const PrimExpr& x) { return as_int(x) == 1; }

		tirx::IndexMap make_iter_fusion_index_map(const std::vector<IterTrait>& traits, const std::vector<IterKind>& kind_order) {
			std::map<IterKind, PrimExpr> fused_iters;
			Array<PrimVar> input_iters;
			for (size_t i = 0; i < traits.size(); ++i) {
				const auto& trait = traits[i];
				PrimVar v_i("i" + std::to_string(i), trait.extent.ty());
				input_iters.push_back(v_i);
				if (trait.kind == IterKind::kIter_T)
					continue;
				if (std::find(kind_order.begin(), kind_order.end(), trait.kind) == kind_order.end())
					throw std::invalid_argument("Unknown iter kind");
				if (auto it = fused_iters.find(trait.kind); it != fused_iters.end())
					it->second = it->second * trait.extent + v_i;
				else
					fused_iters.emplace(trait.kind, v_i);
			}

			Array<PrimExpr> final_indices;
			for (const auto& kind : kind_order) {
				auto it = fused_iters.find(kind);
				final_indices.push_back(it != fused_iters.end() ? it->second : ::tvm::IntImm(traits[0].extent.ty(), 0));
			}

			return tirx::IndexMap(input_iters, final_indices, std::nullopt);
		}

		struct Traits {
			std::vector<IterTrait> A_traits, B_traits, C_traits, block_traits;
		};

		/**
		 * @brief Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K].
		 * @return the traits of A, B, C and the block; std::nullopt if the block is not a GEMM-like block.
		 */
		std::optional<Traits> detect_iter_traits(const s_tir::SBlock& block) {
			if (block->reads.size() != 2 || block->writes.size() != 1)
				return std::nullopt;

			using VarSet = std::unordered_set<const void*>;
			auto get_access_axes = [](const Array<Range>& region) -> std::optional<VarSet> {
				VarSet axes;
				for (const auto& r : region) {
					if (!_is_one(r->extent))
						return std::nullopt; // raise ValueError("Expect elemwise block access")
					for (const auto& v : tirx::UndefinedVars(r->min))
						axes.insert(v.get());
				}
				return axes;
			};

			const auto A_axes = get_access_axes(block->reads[0]->region);
			const auto B_axes = get_access_axes(block->reads[1]->region);
			const auto C_axes = get_access_axes(block->writes[0]->region);
			if (!A_axes || !B_axes || !C_axes)
				return std::nullopt;

			std::unordered_map<const void*, IterTrait> traits;
			for (const auto& iter_var : block->iter_vars) {
				const void* var = iter_var->var.get();
				IterKind kind;
				if (_is_one(iter_var->dom->extent)) {
					kind = IterKind::kIter_T;
				} else if (iter_var->iter_type == tirx::kDataPar) {
					if (A_axes->contains(var) && B_axes->contains(var) && C_axes->contains(var))
						kind = IterKind::kIter_S;
					else if (A_axes->contains(var) && C_axes->contains(var))
						kind = IterKind::kIter_I;
					else if (B_axes->contains(var) && C_axes->contains(var))
						kind = IterKind::kIter_J;
					else
						return std::nullopt;
				} else if (iter_var->iter_type == tirx::kCommReduce) {
					if (A_axes->contains(var) && B_axes->contains(var) && !C_axes->contains(var))
						kind = IterKind::kIter_K;
					else
						return std::nullopt;
				} else {
					return std::nullopt;
				}
				traits.emplace(var, IterTrait{kind, iter_var->dom->extent});
			}

			// A Gemm-kernel requires have I, J and K axes
			std::set<IterKind> kinds;
			for (const auto& [_, t] : traits)
				kinds.insert(t.kind);
			for (IterKind k : {IterKind::kIter_I, IterKind::kIter_J, IterKind::kIter_K})
				if (!kinds.contains(k))
					return std::nullopt;

			Traits result;
			for (const auto& iter_var : block->iter_vars) {
				const void* var = iter_var->var.get();
				if (A_axes->contains(var))
					result.A_traits.push_back(traits.at(var));
				if (B_axes->contains(var))
					result.B_traits.push_back(traits.at(var));
				if (C_axes->contains(var))
					result.C_traits.push_back(traits.at(var));
				result.block_traits.push_back(traits.at(var));
			}
			return result;
		}

		struct IndexMaps {
			tirx::IndexMap matmul_index_map, a_index_map, b_index_map, c_index_map;
		};

		/** @brief Get index maps for the block: (matmul, A, B, C); std::nullopt if not a GEMM-like block. */
		std::optional<IndexMaps> get_index_map(const s_tir::SBlock& block) {
			const auto traits = detect_iter_traits(block);
			if (!traits)
				return std::nullopt;
			using enum IterKind;
			return IndexMaps{
					.matmul_index_map = make_iter_fusion_index_map(traits->block_traits, {kIter_S, kIter_I, kIter_J, kIter_K}),
					.a_index_map = make_iter_fusion_index_map(traits->A_traits, {kIter_S, kIter_I, kIter_K}),
					.b_index_map = make_iter_fusion_index_map(traits->B_traits, {kIter_S, kIter_J, kIter_K}),
					.c_index_map = make_iter_fusion_index_map(traits->C_traits, {kIter_S, kIter_I, kIter_J}),
			};
		}

		std::optional<Array<SBlockRV>> get_reduction_blocks(const Schedule& sch, const Array<SBlockRV>& blocks) {
			// Get the main computation block
			auto iter_types = [&](const SBlockRV& block) {
				std::set<int> types;
				for (const auto& iter_var : sch->Get(block)->iter_vars)
					types.insert(static_cast<int>(iter_var->iter_type));
				return types;
			};
			auto is_reduction = [&](const SBlockRV& block) {
				return iter_types(block) == std::set<int>{tirx::kCommReduce, tirx::kDataPar};
			};
			auto is_spatial = [&](const SBlockRV& block) { return iter_types(block) == std::set<int>{tirx::kDataPar}; };

			// NOTE: We assume there is only one reduction block in the function
			// all blocks are required to be spatial or reduction
			for (const auto& block : blocks)
				if (!is_reduction(block) && !is_spatial(block))
					return std::nullopt;

			// There is only one reduction block
			Array<SBlockRV> reduction_blocks;
			for (const auto& block : blocks)
				if (is_reduction(block))
					reduction_blocks.push_back(block);
			if (reduction_blocks.size() != 1)
				return std::nullopt;

			return reduction_blocks;
		}

		/** @brief Detect In/Out data types for the given block based on the analysis of read/write buffers. */
		std::pair<::tvm::PrimType, ::tvm::PrimType> get_in_out_dtypes(const s_tir::SBlock& block) {
			if (block->reads.empty() || block->writes.empty())
				throw std::logic_error("get_in_out_dtypes");
			return {source_of(block->reads[0])->dtype, source_of(block->writes[0])->dtype};
		}

		int64_t check_sm_version(const std::string& arch) {
			const std::string sm_version = arch.starts_with("sm_") ? arch.substr(3) : arch;
			return !sm_version.empty() && sm_version.find_first_not_of("0123456789") == std::string::npos
						   ? std::stoll(sm_version)
						   : -1;
		}

		/** @brief Configuration for the matmul schedule (Matmul.Config). */
		struct Config {
			int64_t block_size_x = 8;
			int64_t block_size_y = 8;
			int64_t vthread_x = 1;
			int64_t vthread_y = 1;
			int64_t micro_size_x = 4;
			int64_t micro_size_y = 4;
			int64_t micro_size_k = 8;
			int64_t vector_size = 1;
			int64_t unroll = 256; // 0 means no unroll
			bool use_shared = true;
			bool storage_align = false;
			bool inner_x = false;
		};

		Config get_configs(const Target& target) {
			// Get the schedule config for the target
			const std::string kind = target->kind->name;
			if (kind == "cuda" || kind == "rocm")
				return Config{
						.block_size_x = 8, .block_size_y = 16, .vthread_x = 1, .vthread_y = 1, .micro_size_x = 4,
						.micro_size_y = 4, .micro_size_k = 16, .vector_size = 2, .unroll = 256, .use_shared = true,
						.storage_align = true, .inner_x = false
				};
			// (Not ported: the configuration of OpenCL on Android/Adreno.)
			return Config{};
		}
	} // namespace

	namespace {
		Array<PrimExpr> ints(std::initializer_list<int64_t> values) {
			Array<PrimExpr> result;
			for (int64_t v : values)
				result.push_back(::tvm::IntImm::Int32(v));
			return result;
		}
	} // namespace

	std::optional<Schedule>
	MatmulTensorization::apply(const PrimFunc& func, const Target& target, bool /*tunable*/) const {
		// pylint: disable=too-many-locals,missing-docstring
		tensor_intrin::cuda::register_wmma_intrinsics(); // (Python: importing tvm.s_tir.tensor_intrin.cuda)

		if (!is_target_available(target))
			return std::nullopt;
		Schedule sch = schedule_of(func);
		const SBlockRV root_block = analysis::get_root_block(sch);
		const auto blocks = sch->GetChildBlocks(root_block);

		if (func->attrs.defined() && func->attrs->dict.count("dlight.do_not_tensorize"))
			return std::nullopt;

		const auto reduction_blocks = get_reduction_blocks(sch, blocks);
		if (!reduction_blocks)
			return std::nullopt;

		// Start Schedule
		// Step 0. Get schedule config.
		// NOTE: we can analyze the config by the hardware spec in the future

		// tensor core intrinsic size
		const int64_t micro_size_x = 16;
		const int64_t micro_size_y = 16;
		const int64_t micro_size_k = 16;

		const int64_t warp_size = 32;
		const int64_t vector_size = 4;

		const std::vector<std::optional<int64_t>> i_factors{std::nullopt, 1, 4, 2}, j_factors{1, std::nullopt, 4, 2},
				k_factors{std::nullopt, 4};
		const int64_t num_ty = *i_factors[2] * *j_factors[2];
		const int64_t x_pad_factor = *i_factors[2] * *i_factors[3];
		const int64_t y_pad_factor = *j_factors[2] * *j_factors[3];
		const int64_t k_pad_factor = *k_factors[1];

		// Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
		const SBlockRV main_block = (*reduction_blocks)[0];
		const SBlockRV reindex_a = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kRead);
		const SBlockRV reindex_b = sch->ReIndex(main_block, 1, s_tir::BufferIndexType::kRead);
		const SBlockRV reindex_c = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kWrite);

		const auto index_maps = get_index_map(sch->Get(main_block));
		if (!index_maps)
			throw std::logic_error("MatmulTensorization: no index maps after reindexing");
		sch->TransformLayout(reindex_a, 0, s_tir::BufferIndexType::kWrite, index_maps->a_index_map);
		sch->TransformLayout(reindex_b, 0, s_tir::BufferIndexType::kWrite, index_maps->b_index_map);
		sch->TransformLayout(reindex_c, 0, s_tir::BufferIndexType::kRead, index_maps->c_index_map);
		sch->TransformBlockLayout(main_block, index_maps->matmul_index_map);

		// Step 2. Padding for dynamic shape kernels
		sch->PadEinsum(
				main_block,
				{1, micro_size_x * x_pad_factor, micro_size_y * y_pad_factor, micro_size_k * k_pad_factor}
		);

		// Step 3. Schedule matmul to use tensor core
		const SBlockRV block = main_block;

		auto loops = sch->GetLoops(block);
		const LoopRV batch = loops[0];
		auto split = sch->Split(loops[1], factors({std::nullopt, micro_size_x}));
		LoopRV i = split[0], i_inner = split[1];
		split = sch->Split(loops[2], factors({std::nullopt, micro_size_y}));
		LoopRV j = split[0], j_inner = split[1];
		split = sch->Split(loops[3], factors({std::nullopt, micro_size_k}));
		LoopRV k = split[0], k_inner = split[1];

		sch->Reorder({i, j, k, i_inner, j_inner, k_inner});

		const SBlockRV block_inner = block;
		const SBlockRV block_outer = sch->Blockize(i_inner);

		auto to_factors = [](const std::vector<std::optional<int64_t>>& f) {
			Array<ffi::Optional<PrimExpr>> result;
			for (const auto& v : f)
				result.push_back(v ? ffi::Optional<PrimExpr>(::tvm::IntImm::Int32(*v)) : ffi::Optional<PrimExpr>());
			return result;
		};
		split = sch->Split(i, to_factors(i_factors));
		const LoopRV i0 = split[0], i1 = split[1], i2 = split[2], i3 = split[3];
		split = sch->Split(j, to_factors(j_factors));
		const LoopRV j0 = split[0], j1 = split[1], j2 = split[2], j3 = split[3];
		split = sch->Split(k, to_factors(k_factors));
		const LoopRV k0 = split[0], k1 = split[1];
		sch->Annotate(k0, "software_pipeline_order", ints({0, 3, 1, 4, 5, 2, 6}));
		sch->Annotate(k0, "software_pipeline_stage", ints({0, 0, 0, 0, 0, 1, 1}));
		sch->Annotate(k1, "software_pipeline_order", ints({0, 1, 2}));
		sch->Annotate(k1, "software_pipeline_stage", ints({0, 0, 1}));

		sch->Reorder({i0, j0, i1, j1, j2, i2, k0, k1, i3, j3});

		const LoopRV block_idx = sch->Fuse({i0, j0});
		const LoopRV block_idy = sch->Fuse({i1, j1});
		const LoopRV thread_idy = sch->Fuse({j2, i2});
		sch->Bind(batch, "blockIdx.z");
		sch->Bind(block_idx, "blockIdx.x");
		sch->Bind(block_idy, "blockIdx.y");
		sch->Bind(thread_idy, "threadIdx.y");

		auto fetch_to_shared = [&](const SBlockRV& block, int idx, int ndim) {
			const SBlockRV block_read = sch->CacheRead(block, idx, "shared.dyn");
			sch->ComputeAt(block_read, k0, /*preserve_unit_loops=*/false);
			const auto read_loops = sch->GetLoops(block_read);
			const LoopRV fused = sch->Fuse(Array<LoopRV>(read_loops.end() - ndim, read_loops.end()));

			const auto parts = sch->Split(fused, factors({std::nullopt, num_ty, warp_size, vector_size}));
			const LoopRV f_1 = parts[1], f_2 = parts[2], f_3 = parts[3];

			sch->Bind(f_2, "threadIdx.x");
			sch->Bind(f_1, "threadIdx.y");
			sch->Vectorize(f_3);

			sch->StorageAlign(block_read, 0, /*axis=*/-2, /*factor=*/16, /*offset=*/8);
			sch->Annotate(block_read, "tirx.manifest_shared_memory_local_stage", ::tvm::IntImm::Int32(1));
			sch->Annotate(block_read, "double_buffer_scope", ::tvm::IntImm::Int32(0));
			return block_read;
		};

		const SBlockRV a_g2s = fetch_to_shared(block_outer, 0, 2);
		const SBlockRV b_g2s = fetch_to_shared(block_outer, 1, 2);

		auto_inline_producers(sch, a_g2s);
		auto_inline_producers(sch, b_g2s);

		// create read cache to load matrix from shared memory to wmma fragments
		const SBlockRV A_mat = sch->CacheRead(block_outer, 0, "wmma.matrix_a");
		const SBlockRV B_mat = sch->CacheRead(block_outer, 1, "wmma.matrix_b");
		sch->ComputeAt(A_mat, k1, /*preserve_unit_loops=*/false);
		sch->ComputeAt(B_mat, k1, /*preserve_unit_loops=*/false);

		// create write cache to store matrix from wmma fragments to shared memory and global memory
		const SBlockRV accumulator_shared_to_global = sch->CacheWrite(block_outer, 0, "shared.dyn");
		sch->StorageAlign(accumulator_shared_to_global, 0, -2, 16, 4);

		const SBlockRV store = sch->CacheWrite(block_outer, 0, "wmma.accumulator");
		sch->ReverseComputeAt(store, thread_idy, /*preserve_unit_loops=*/false);
		sch->ReverseComputeAt(accumulator_shared_to_global, thread_idy, /*preserve_unit_loops=*/false);

		// split the store loop to match hardware intrinsic pattern
		auto store_loops = sch->GetLoops(store);
		auto si = sch->Split(store_loops[store_loops.size() - 2], factors({std::nullopt, 16}));
		auto sj = sch->Split(store_loops[store_loops.size() - 1], factors({std::nullopt, 16}));
		sch->Reorder({si[0], sj[0], si[1], sj[1]});

		const SBlockRV block_init_c = sch->DecomposeReduction(block_outer, k0);
		const SBlockRV block_init_c_inner = sch->GetChildBlocks(block_init_c)[0];

		// Tensorization by hardware intrinsics
		auto intrin_group = tensor_intrin::cuda::get_wmma_intrin_group(
				"shared.dyn", "shared.dyn", "float16", "float32", /*trans_b=*/true
		);

		try {
			for (const auto& [mat, intrin] : {std::pair{A_mat, "load_a"}, std::pair{B_mat, "load_b"}}) {
				const auto mat_loops = sch->GetLoops(mat);
				const auto mi = sch->Split(mat_loops[mat_loops.size() - 2], factors({std::nullopt, 16}));
				const auto mj = sch->Split(mat_loops[mat_loops.size() - 1], factors({std::nullopt, 16}));
				sch->Reorder({mi[0], mj[0], mi[1], mj[1]});
				sch->Unroll(mi[0]);
				sch->Unroll(mj[0]);
				sch->Tensorize(mi[1], intrin_group.at(intrin));
			}
		} catch (const std::exception&) {
			return std::nullopt;
		}

		bool tensorize_success = false;

		auto tensorize_init_store_compute = [&] {
			const auto init_loops = sch->GetLoops(block_init_c_inner);
			sch->Tensorize(init_loops[init_loops.size() - 2], intrin_group.at("init"));
			const auto s_loops = sch->GetLoops(store);
			sch->Tensorize(s_loops[s_loops.size() - 2], intrin_group.at("store"));
			const auto c_loops = sch->GetLoops(block_inner);
			sch->Tensorize(c_loops[c_loops.size() - 3], intrin_group.at("compute"));
		};

		try {
			tensorize_init_store_compute();
			tensorize_success = true;
		} catch (const std::exception&) {
			intrin_group = tensor_intrin::cuda::get_wmma_intrin_group(
					"shared.dyn", "shared.dyn", "float16", "float16", /*trans_b=*/true
			);
		}

		if (!tensorize_success) {
			try {
				tensorize_init_store_compute();
				tensorize_success = true;
			} catch (const std::exception&) {
				return std::nullopt;
			}
		}
		auto_inline_consumer_chain(sch, accumulator_shared_to_global);

		const auto acc_loops = sch->GetLoops(accumulator_shared_to_global);
		const LoopRV fused = sch->Fuse(Array<LoopRV>(acc_loops.end() - 2, acc_loops.end()));
		const auto parts = sch->Split(fused, factors({std::nullopt, warp_size, vector_size}));
		sch->Bind(parts[1], "threadIdx.x");
		sch->Vectorize(parts[2]);

		if (tensorize_success)
			return sch;
		return std::nullopt;
	}

	std::optional<Schedule> Matmul::apply(const PrimFunc& func, const Target& target, bool tunable) const {
		// pylint: disable=too-many-locals,missing-docstring
		if (!is_target_available(target))
			return std::nullopt;
		Schedule sch = schedule_of(func);
		const Config config = get_configs(target);
		const SBlockRV root_block = analysis::get_root_block(sch);
		const auto blocks = sch->GetChildBlocks(root_block);

		const auto reduction_blocks = get_reduction_blocks(sch, blocks);
		if (!reduction_blocks)
			return std::nullopt;

		const SBlockRV main_block = (*reduction_blocks)[0];
		s_tir::SBlock block_stmt = sch->Get(main_block);
		if (!get_index_map(block_stmt))
			return std::nullopt;

		// (Not ported: sch_outer_reduction for OpenCL on Android/Adreno.)

		// Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
		const SBlockRV reindex_a = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kRead);
		const SBlockRV reindex_b = sch->ReIndex(main_block, 1, s_tir::BufferIndexType::kRead);
		const SBlockRV reindex_c = sch->ReIndex(main_block, 0, s_tir::BufferIndexType::kWrite);

		const auto index_maps = get_index_map(sch->Get(main_block));
		if (!index_maps)
			throw std::logic_error("Matmul: no index maps after reindexing");
		sch->TransformLayout(reindex_a, 0, s_tir::BufferIndexType::kWrite, index_maps->a_index_map);
		sch->TransformLayout(reindex_b, 0, s_tir::BufferIndexType::kWrite, index_maps->b_index_map);
		sch->TransformLayout(reindex_c, 0, s_tir::BufferIndexType::kRead, index_maps->c_index_map);
		sch->TransformBlockLayout(main_block, index_maps->matmul_index_map);

		// Try to tensorize the init, store and compute block with f16 or f32 intrinsics
		const int64_t minimal_tensorize_threshold = 64;
		block_stmt = sch->Get(main_block);
		const auto arch = target->GetAttr<ffi::String>("arch");
		if (target->kind->name == "cuda" && check_sm_version(arch ? std::string(arch.value()) : "") >= 70) {
			bool apply_tensorization = true;
			// the batch dimension is not taken into consideration.
			for (size_t i = 1; i < block_stmt->iter_vars.size(); ++i) {
				const auto extent = as_int(block_stmt->iter_vars[i]->dom->extent);
				if (extent && *extent <= minimal_tensorize_threshold)
					apply_tensorization = false;
			}
			if (apply_tensorization) {
				const auto [in_dtype, out_dtype] = get_in_out_dtypes(block_stmt);
				std::optional<Schedule> tensorize_sch;
				if (in_dtype == ::tvm::PrimType::Int(8) && out_dtype == ::tvm::PrimType::Int(32)) {
					// (Not ported: MatmulInt8Tensorization.)
				} else if (in_dtype == ::tvm::PrimType::Float(16) &&
						   (out_dtype == ::tvm::PrimType::Float(16) || out_dtype == ::tvm::PrimType::Float(32))) {
					tensorize_sch = MatmulTensorization().apply(func, target, tunable);
				}
				if (tensorize_sch)
					return tensorize_sch;
			}
		}
		// (Not ported: MetalMatmul.)

		// Step 2. Get schedule config.
		const int64_t y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y;
		const int64_t x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x;
		LoopRV batch, x, y, k;
		if (config.inner_x) {
			sch->PadEinsum(main_block, {1, y_kernel_size, x_kernel_size, config.micro_size_k});
			const auto loops = sch->GetLoops(main_block);
			batch = loops[0];
			y = loops[1];
			x = loops[2];
			k = loops[3];
		} else {
			sch->PadEinsum(main_block, {1, x_kernel_size, y_kernel_size, config.micro_size_k});
			const auto loops = sch->GetLoops(main_block);
			batch = loops[0];
			x = loops[1];
			y = loops[2];
			k = loops[3];
		}

		// Step 3. Schedule matmul
		auto split = sch->Split(
				y, factors({std::nullopt, config.vthread_y, config.block_size_y, config.micro_size_y})
		);
		LoopRV by = split[0];
		const LoopRV vy = split[1], ty = split[2], yi = split[3];
		split = sch->Split(x, factors({std::nullopt, config.vthread_x, config.block_size_x, config.micro_size_x}));
		const LoopRV bx = split[0], vx = split[1], tx = split[2], xi = split[3];
		split = sch->Split(k, factors({std::nullopt, config.micro_size_k}));
		const LoopRV ko = split[0], ki = split[1];
		Array<LoopRV> reordered_loops{by, bx, vy, vx, ty, tx, ko, ki};
		if (config.inner_x) {
			reordered_loops.push_back(yi);
			reordered_loops.push_back(xi);
		} else {
			reordered_loops.push_back(xi);
			reordered_loops.push_back(yi);
		}
		sch->Reorder(reordered_loops);
		by = sch->Fuse({batch, by});
		sch->Bind(bx, "blockIdx.x");
		sch->Bind(by, "blockIdx.y");
		sch->Bind(vy, "vthread.y");
		sch->Bind(vx, "vthread.x");
		sch->Bind(ty, "threadIdx.y");
		sch->Bind(tx, "threadIdx.x");
		const int64_t inner_loop = config.inner_x ? config.micro_size_x : config.micro_size_y;
		if (inner_loop % config.vector_size == 0) {
			const LoopRV v = sch->Split(reordered_loops.back(), factors({std::nullopt, config.vector_size}))[1];
			sch->Vectorize(v);
		}

		if (config.unroll > 0) {
			sch->Annotate(tx, "pragma_auto_unroll_max_step", ::tvm::IntImm::Int32(config.unroll));
			sch->Annotate(tx, "pragma_unroll_explicit", ::tvm::IntImm::Int32(1));
		}

		const SBlockRV l2g = sch->CacheWrite(main_block, 0, "local");
		sch->ReverseComputeAt(l2g, tx, /*preserve_unit_loops=*/true);
		if (config.micro_size_x % config.vector_size == 0) {
			const LoopRV v =
					sch->Split(sch->GetLoops(l2g).back(), factors({std::nullopt, config.vector_size}))[1];
			sch->Vectorize(v);
		}

		if (config.use_shared) {
			auto _cooperative_fetch = [&](int index, int64_t vec_len) {
				const SBlockRV block = sch->CacheRead(main_block, index, "shared");
				const size_t num_loops = sch->GetLoops(block).size();
				sch->ComputeAt(block, ko, /*preserve_unit_loops=*/true);
				const auto all_loops = sch->GetLoops(block);
				const Array<LoopRV> loops(all_loops.end() - static_cast<std::ptrdiff_t>(num_loops), all_loops.end());
				const auto parts =
						sch->Split(sch->Fuse(loops), factors({config.block_size_y, config.block_size_x, std::nullopt, vec_len}));
				sch->Vectorize(parts[3]);
				sch->Bind(parts[0], "threadIdx.y");
				sch->Bind(parts[1], "threadIdx.x");
				if (config.storage_align)
					sch->StorageAlign(block, 0, /*axis=*/1, /*factor=*/8, /*offset=*/static_cast<int>(vec_len));
				return block;
			};

			const SBlockRV a_g2s = _cooperative_fetch(0, config.vector_size);
			const SBlockRV b_g2s = _cooperative_fetch(1, config.vector_size);

			auto_inline_producers(sch, a_g2s);
			auto_inline_producers(sch, b_g2s);
		} else {
			auto_inline_producers(sch, main_block);
		}

		auto_inline_consumer_chain(sch, l2g);
		sch->DecomposeReduction(main_block, ko);
		return sch;
	}

} // namespace mdl::tvm_ported::dlight::gpu
