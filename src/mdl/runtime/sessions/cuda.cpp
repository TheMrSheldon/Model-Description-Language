// The CUDA device sessions of compiled models (docs/technical/runtime.md): each session runs the calls on its own
// stream and replays them as CUDA graphs. The bindings find the session functions as TVM FFI globals
// `mdl.session.2.{create,call,free}` (2: kDLCUDA) when this library is loaded; otherwise they call directly.
//
// A call of a function with given arguments (the key: data pointers, shapes, strides, element types, scalars) runs
// directly the first time (a warm-up: cuBLASLt autotuning, plan building, and allocations cannot be captured), is
// captured as a graph the second time, and is replayed afterwards. Capture uses cudaStreamCaptureModeThreadLocal, in
// which a synchronizing call (e.g. a copy to the host) fails instead of being run once: such a call is not captured,
// and its key runs directly from then on. MDL_CUDA_GRAPHS=0 disables the graphs (the session's stream remains).
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {
	constexpr size_t kMaxKeys = 32; ///< keys (graphs, warm-ups, failed captures) a session keeps

	using RawFunction = int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

	bool graphsEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("MDL_CUDA_GRAPHS");
			return value == nullptr || std::string_view(value) != "0";
		}();
		return enabled;
	}

	const DLTensor* tensorOf(const TVMFFIAny& arg) {
		if (arg.type_index == kTVMFFIDLTensorPtr)
			return static_cast<const DLTensor*>(arg.v_ptr);
		if (arg.type_index == kTVMFFITensor)
			return TVMFFITensorGetDLTensorPtr(arg.v_obj);
		return nullptr;
	}

	/** @brief What a graph replays: the function and its arguments (as their values, or as the tensors' layouts). */
	std::vector<int64_t> keyOf(const TVMFFIAny& fn, const TVMFFIAny* args, int32_t n) {
		std::vector<int64_t> key{fn.v_int64};
		for (int32_t i = 0; i < n; ++i) {
			key.push_back(args[i].type_index);
			const DLTensor* t = tensorOf(args[i]);
			if (t == nullptr) {
				key.push_back(args[i].v_int64);
				continue;
			}
			uint32_t dtype = 0;
			std::memcpy(&dtype, &t->dtype, sizeof(dtype));
			key.insert(
					key.end(), {reinterpret_cast<int64_t>(t->data) + static_cast<int64_t>(t->byte_offset), t->ndim,
								static_cast<int64_t>(dtype), t->device.device_type, t->device.device_id}
			);
			key.insert(key.end(), t->shape, t->shape + t->ndim);
			if (t->strides != nullptr)
				key.insert(key.end(), t->strides, t->strides + t->ndim);
		}
		return key;
	}

	struct KeyHash {
		size_t operator()(const std::vector<int64_t>& key) const {
			uint64_t h = 1469598103934665603ULL; // FNV-1a
			for (int64_t v : key)
				h = (h ^ static_cast<uint64_t>(v)) * 1099511628211ULL;
			return static_cast<size_t>(h);
		}
	};

	/** @brief Clears the error that a failed call raised (in the TVM FFI and in CUDA). */
	void clearErrors() {
		TVMFFIObjectHandle error = nullptr;
		TVMFFIErrorMoveFromRaised(&error);
		if (error != nullptr)
			TVMFFIObjectDecRef(error);
		(void)cudaGetLastError();
	}

	class Session {
	public:
		explicit Session(int device) : device(device) {
			int previous = 0;
			cudaGetDevice(&previous);
			cudaSetDevice(device);
			// (blocking: ordered with the legacy default stream, on which callers usually prepare their tensors)
			const cudaError_t status = cudaStreamCreate(&stream);
			cudaSetDevice(previous);
			if (status != cudaSuccess)
				throw status;
		}
		Session(const Session&) = delete;
		Session& operator=(const Session&) = delete;
		~Session() {
			for (auto& [_, entry] : entries)
				if (entry.graph != nullptr)
					cudaGraphExecDestroy(entry.graph);
			cudaStreamDestroy(stream);
		}

		/** @brief Calls @p fn (a function pointer or a TVM FFI function) with @p args on the session's stream. */
		int call(const TVMFFIAny& fn, TVMFFIAny* args, int32_t n, TVMFFIAny* result) {
			TVMFFIStreamHandle previous = nullptr;
			TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous);
			const int status = dispatch(fn, args, n, result);
			TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr);
			return status;
		}

	private:
		struct Entry {
			int calls = 0;
			bool failed = false; ///< (its capture failed: runs directly)
			cudaGraphExec_t graph = nullptr;
			uint64_t used = 0;
		};

		int device;
		cudaStream_t stream = nullptr;
		std::unordered_map<std::vector<int64_t>, Entry, KeyHash> entries;
		uint64_t clock = 0;

		static int invoke(const TVMFFIAny& fn, TVMFFIAny* args, int32_t n, TVMFFIAny* result) {
			if (fn.type_index == kTVMFFIOpaquePtr)
				return reinterpret_cast<RawFunction>(fn.v_ptr)(nullptr, args, n, result);
			return TVMFFIFunctionCall(fn.v_obj, args, n, result);
		}

		int dispatch(const TVMFFIAny& fn, TVMFFIAny* args, int32_t n, TVMFFIAny* result) {
			if (!graphsEnabled())
				return invoke(fn, args, n, result);
			Entry& entry = find(keyOf(fn, args, n));
			entry.used = ++clock;
			if (entry.graph != nullptr)
				return cudaGraphLaunch(entry.graph, stream) == cudaSuccess ? 0 : fail("cudaGraphLaunch failed");
			if (entry.failed || entry.calls++ == 0)
				return invoke(fn, args, n, result); // (the warm-up, or not capturable)
			if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
				clearErrors();
				entry.failed = true;
				return invoke(fn, args, n, result);
			}
			const int status = invoke(fn, args, n, result);
			cudaGraph_t graph = nullptr;
			const cudaError_t captured = cudaStreamEndCapture(stream, &graph);
			if (status == 0 && captured == cudaSuccess && graph != nullptr &&
				cudaGraphInstantiate(&entry.graph, graph, 0) == cudaSuccess) {
				cudaGraphDestroy(graph);
				return cudaGraphLaunch(entry.graph, stream) == cudaSuccess ? 0 : fail("cudaGraphLaunch failed");
			}
			// Not capturable (or the call failed): run it directly, now and from then on (a failure is then raised).
			if (graph != nullptr)
				cudaGraphDestroy(graph);
			entry.graph = nullptr;
			clearErrors();
			entry.failed = true;
			return invoke(fn, args, n, result);
		}

		/** @brief The entry of @p key, evicting the least recently used one if the session keeps too many. */
		Entry& find(std::vector<int64_t> key) {
			if (auto it = entries.find(key); it != entries.end())
				return it->second;
			if (entries.size() >= kMaxKeys) {
				auto oldest = entries.begin();
				for (auto it = entries.begin(); it != entries.end(); ++it)
					if (it->second.used < oldest->second.used)
						oldest = it;
				if (oldest->second.graph != nullptr)
					cudaGraphExecDestroy(oldest->second.graph);
				entries.erase(oldest);
			}
			return entries[std::move(key)];
		}

		static int fail(const char* message) {
			TVMFFIErrorSetRaisedFromCStr("RuntimeError", message);
			return -1;
		}
	};

	TVMFFIAny opaque(void* p) {
		TVMFFIAny any{};
		any.type_index = kTVMFFIOpaquePtr;
		any.v_ptr = p;
		return any;
	}

	/** @brief `create(device_id) -> session`. */
	int create(void*, const TVMFFIAny* args, int32_t n, TVMFFIAny* result) {
		if (n != 1 || args[0].type_index != kTVMFFIInt) {
			TVMFFIErrorSetRaisedFromCStr("TypeError", "mdl.session.2.create(device_id)");
			return -1;
		}
		try {
			*result = opaque(new Session(static_cast<int>(args[0].v_int64)));
			return 0;
		} catch (cudaError_t status) {
			TVMFFIErrorSetRaisedFromCStr("RuntimeError", cudaGetErrorString(status));
			return -1;
		}
	}

	/** @brief `call(session, function, args...)`: the function (a pointer or a TVM FFI function) of args. */
	int call(void*, const TVMFFIAny* args, int32_t n, TVMFFIAny* result) {
		if (n < 2 || args[0].type_index != kTVMFFIOpaquePtr) {
			TVMFFIErrorSetRaisedFromCStr("TypeError", "mdl.session.2.call(session, function, args...)");
			return -1;
		}
		// (TVMFFIFunctionCall takes the arguments as mutable)
		std::vector<TVMFFIAny> rest(args + 2, args + n);
		return static_cast<Session*>(args[0].v_ptr)->call(args[1], rest.data(), n - 2, result);
	}

	/** @brief `free(session)`. */
	int free(void*, const TVMFFIAny* args, int32_t n, TVMFFIAny*) {
		if (n == 1 && args[0].type_index == kTVMFFIOpaquePtr)
			delete static_cast<Session*>(args[0].v_ptr);
		return 0;
	}

	void registerGlobal(const char* name, TVMFFISafeCallType fn) {
		TVMFFIObjectHandle handle = nullptr;
		if (TVMFFIFunctionCreate(nullptr, fn, nullptr, &handle) != 0)
			return;
		const TVMFFIByteArray bytes{name, std::strlen(name)};
		TVMFFIFunctionSetGlobal(&bytes, handle, /*allow_override=*/1);
		TVMFFIObjectDecRef(handle);
	}

	const bool registered = [] {
		registerGlobal("mdl.session.2.create", create);
		registerGlobal("mdl.session.2.call", call);
		registerGlobal("mdl.session.2.free", free);
		return true;
	}();
} // namespace
