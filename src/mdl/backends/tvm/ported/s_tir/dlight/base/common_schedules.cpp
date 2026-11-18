// Translated from https://github.com/apache/tvm/blob/7b5acb32bd271342a24130f4ef439ff6c2e52c6c/python/tvm/s_tir/dlight/base/common_schedules.py
// (conventions: src/mdl/backends/tvm/ported/README.md)
#include "./common_schedules.hpp"

#include <exception>
#include <functional>

namespace mdl::tvm_ported::dlight {

	std::vector<analysis::SBlockInfo> try_inline(const Schedule& sch, std::vector<analysis::SBlockInfo> blocks) {
		auto trial = [&](const std::function<void(const SBlockRV&)>& func) -> std::optional<size_t> {
			for (size_t i = 0; i < blocks.size(); ++i) {
				try {
					func(blocks[i].block_rv);
				} catch (const std::exception&) {
					continue;
				}
				return i;
			}
			return std::nullopt;
		};
		while (true) {
			auto i = trial([&](const SBlockRV& b) { sch->ComputeInline(b); });
			if (!i)
				i = trial([&](const SBlockRV& b) { sch->ReverseComputeInline(b); });
			if (!i)
				break;
			blocks.erase(blocks.begin() + static_cast<std::ptrdiff_t>(*i));
		}
		return blocks;
	}

	std::optional<std::vector<analysis::SBlockInfo>>
	try_inline_contiguous_spatial(const Schedule& sch, const std::optional<std::vector<analysis::SBlockInfo>>& block_infos) {
		if (!block_infos)
			return std::nullopt;
		std::vector<analysis::SBlockInfo> results, spatial_blocks;
		for (const auto& block : *block_infos) {
			if (block.is_injective()) {
				spatial_blocks.push_back(block);
			} else if (!spatial_blocks.empty()) {
				for (auto& b : try_inline(sch, spatial_blocks))
					results.push_back(b);
				results.push_back(block);
				spatial_blocks.clear();
			} else {
				results.push_back(block);
			}
		}
		if (!spatial_blocks.empty())
			for (auto& b : try_inline(sch, spatial_blocks))
				results.push_back(b);
		return results;
	}

} // namespace mdl::tvm_ported::dlight
