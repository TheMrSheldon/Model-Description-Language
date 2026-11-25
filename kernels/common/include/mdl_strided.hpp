#ifndef MDL_STRIDED_HPP
#define MDL_STRIDED_HPP

/**
 * @file
 * @brief Strided tensor views of the kernel packages' C ABIs (header-only, shared by the packages' libraries).
 * @details A compiled model passes every tensor as a view: its rank, shape, and strides (in elements) in the logical
 * layout of the operation, so permuted operands (e.g. the heads of attention) are read and written in place. The
 * helpers here bring views into the forms that the libraries take (e.g. at most two batch dimensions), merging
 * dimensions where the layout allows it and iterating over the others.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

namespace mdl::kernels {

	/** @brief A strided view: dimension i has size shape[i] and stride strides[i] (in elements). */
	struct View {
		int64_t rank;
		const int64_t* shape;
		const int64_t* strides;
	};

	/**
	 * @brief A batched matrix product C[i, j] = A[i, j] @ B[i, j] over two batch dimensions i < b0, j < b1, with the
	 * strides of A [M, K], B [K, N] (a batch stride of 0 broadcasts), and C [M, N].
	 */
	struct Gemm {
		int64_t b0 = 1, b1 = 1, M = 0, K = 0, N = 0;
		int64_t a_s0 = 0, a_s1 = 0, a_sM = 0, a_sK = 0;
		int64_t b_s0 = 0, b_s1 = 0, b_sK = 0, b_sN = 0;
		int64_t c_s0 = 0, c_s1 = 0, c_sM = 0, c_sN = 0;
	};

	/**
	 * @brief Calls @p f(gemm, aOffset, bOffset, cOffset) for the matrix product of the views @p a [..., M, K],
	 * @p b [..., K, N] (batch dimensions broadcast like numpy), and @p c [..., M, N] (the broadcast batch dimensions).
	 * @details Batch dimensions over which B does not vary and that continue M in A and C become part of M (one larger
	 * product, e.g. a linear layer over [batch, seq]); adjacent batch dimensions that are one dimension in all operands
	 * are merged. Up to @p maxBatch remaining batch dimensions are those of the Gemm; @p f is called once per element
	 * of the others (with the offsets of the slices, in elements).
	 * @return the first nonzero result of @p f, or 0.
	 */
	template <typename F>
	int forEachGemm(View a, View b, View c, int maxBatch, F&& f) {
		struct Dim {
			int64_t size, a, b, c;
		};
		const int64_t ra = a.rank, rb = b.rank, rc = c.rank;
		if (ra < 2 || rb < 2 || rc < 2)
			return 100;
		Gemm g;
		g.M = a.shape[ra - 2], g.K = a.shape[ra - 1], g.N = b.shape[rb - 1];
		g.a_sM = a.strides[ra - 2], g.a_sK = a.strides[ra - 1];
		g.b_sK = b.strides[rb - 2], g.b_sN = b.strides[rb - 1];
		g.c_sM = c.strides[rc - 2], g.c_sN = c.strides[rc - 1];
		std::vector<Dim> batch;
		for (int64_t i = 0; i + 2 < rc; ++i) {
			auto stride = [&](const View& v, int64_t r) {
				const int64_t j = i - (rc - r);
				return j >= 0 && v.shape[j] != 1 ? v.strides[j] : int64_t{0};
			};
			if (c.shape[i] != 1)
				batch.push_back({c.shape[i], stride(a, ra), stride(b, rb), c.strides[i]});
		}
		// Batch dimensions that continue M.
		while (!batch.empty() && batch.back().b == 0 && batch.back().a == g.a_sM * g.M &&
			   batch.back().c == g.c_sM * g.M) {
			g.M *= batch.back().size;
			batch.pop_back();
		}
		// Adjacent batch dimensions that are one dimension in all operands.
		for (size_t i = batch.size(); i-- > 1;) {
			const Dim outer = batch[i - 1], inner = batch[i];
			if (outer.a == inner.a * inner.size && outer.b == inner.b * inner.size && outer.c == inner.c * inner.size) {
				batch[i - 1] = {outer.size * inner.size, inner.a, inner.b, inner.c};
				batch.erase(batch.begin() + static_cast<std::ptrdiff_t>(i));
			}
		}
		// The last (at most) maxBatch dimensions are the Gemm's; the others are iterated.
		const size_t inner = std::min(batch.size(), static_cast<size_t>(maxBatch < 0 ? 0 : maxBatch));
		const size_t outer = batch.size() - inner;
		if (inner >= 1) {
			const Dim& d = batch[batch.size() - 1];
			g.b1 = d.size, g.a_s1 = d.a, g.b_s1 = d.b, g.c_s1 = d.c;
		}
		if (inner >= 2) {
			const Dim& d = batch[batch.size() - 2];
			g.b0 = d.size, g.a_s0 = d.a, g.b_s0 = d.b, g.c_s0 = d.c;
		}
		std::vector<int64_t> index(outer, 0);
		while (true) {
			int64_t ao = 0, bo = 0, co = 0;
			for (size_t i = 0; i < outer; ++i)
				ao += index[i] * batch[i].a, bo += index[i] * batch[i].b, co += index[i] * batch[i].c;
			if (const int r = f(g, ao, bo, co); r != 0)
				return r;
			size_t i = outer;
			while (i > 0 && ++index[i - 1] == batch[i - 1].size)
				index[--i] = 0;
			if (i == 0)
				return 0;
		}
	}

	/**
	 * @brief Calls @p f(b0, b1, strides, offsets) for views of rank @p rank with two batch dimensions: a rank of 3 gets
	 * an outer dimension of size 1, larger ranks are iterated over their leading dimensions.
	 * @param views the views (all of the same rank and batch dimensions); @p f receives per view the strides of
	 * [b0, b1, seq, dim] and the offset of the slice.
	 */
	template <typename F>
	int forEachBatch2(const std::vector<View>& views, F&& f) {
		const int64_t rank = views.front().rank;
		if (rank < 2)
			return 100;
		const int64_t outer = rank > 4 ? rank - 4 : 0;
		std::vector<std::vector<int64_t>> strides(views.size());
		auto at = [&](const View& v, int64_t i) { return i >= 0 ? v.strides[i] : int64_t{0}; };
		const int64_t b0 = rank >= 4 ? views.front().shape[rank - 4] : 1;
		const int64_t b1 = rank >= 3 ? views.front().shape[rank - 3] : 1;
		for (size_t t = 0; t < views.size(); ++t)
			strides[t] = {at(views[t], rank - 4), at(views[t], rank - 3), views[t].strides[rank - 2], views[t].strides[rank - 1]};
		std::vector<int64_t> index(static_cast<size_t>(outer), 0);
		std::vector<int64_t> offsets(views.size());
		while (true) {
			for (size_t t = 0; t < views.size(); ++t) {
				offsets[t] = 0;
				for (int64_t i = 0; i < outer; ++i)
					offsets[t] += index[i] * views[t].strides[i];
			}
			if (const int r = f(b0, b1, strides, offsets); r != 0)
				return r;
			int64_t i = outer;
			while (i > 0 && ++index[i - 1] == views.front().shape[i - 1])
				index[--i] = 0;
			if (i == 0)
				return 0;
		}
	}

} // namespace mdl::kernels

#endif
