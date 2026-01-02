#ifndef MDL_UTILS_OVERLOAD_HPP
#define MDL_UTILS_OVERLOAD_HPP

namespace mdl {
	template <class... Ts>
	struct overloaded : Ts... {
		using Ts::operator()...;
	};
} // namespace mdl

#endif