#ifndef MDL_COMPUTEGRAPH_OBJECT_HPP
#define MDL_COMPUTEGRAPH_OBJECT_HPP

#include "./datatype.hpp"

namespace mdl {
	struct ObjectRef final {
		mdl::DataType type;
		/** \todo how to handle value? **/
	};
} // namespace mdl

#endif