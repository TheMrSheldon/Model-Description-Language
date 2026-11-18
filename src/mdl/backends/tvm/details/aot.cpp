#include "./aot.hpp"
#include "./tvm_compat.hpp"

#include <tvm/te/operation.h>
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/type.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	using ::tvm::Expr;
	using ::tvm::GlobalVar;
	using ::tvm::IntImm;
	using ::tvm::IRModule;
	using ::tvm::PrimExpr;
	using ::tvm::PrimVar;
	using ::tvm::StringImm;
	using ::tvm::ffi::Array;
	namespace relax = ::tvm::relax;
	namespace tirx = ::tvm::tirx;
	using DataType = ::tvm::PrimType;

	namespace {
		/** @brief The TIR representation of a Relax value. */
		struct Value {
			enum class Kind { None, Tensor, Prim, Shape, Tuple } kind = Kind::None;
			tirx::BufferVar buffer;
			PrimExpr prim;
			Array<PrimExpr> shape;
			std::vector<Value> fields;

			static Value tensor(tirx::BufferVar b) { return Value{.kind = Kind::Tensor, .buffer = std::move(b)}; }
			static Value primitive(PrimExpr p) { return Value{.kind = Kind::Prim, .prim = std::move(p)}; }
			static Value shapeOf(Array<PrimExpr> s) { return Value{.kind = Kind::Shape, .shape = std::move(s)}; }
		};

		[[noreturn]] void unsupported(const std::string& what) {
			throw std::runtime_error("mdl AOT lowering: unsupported " + what);
		}

		const relax::TensorTypeNode* tensorInfo(const ::tvm::Type& type) { return type.as<relax::TensorTypeNode>(); }

		/** @brief A zero of type @p dtype. */
		PrimExpr zero(DataType dtype) { return ::tvm::prim::MakeConst(dtype, 0); }

		/**
		 * @brief The element type marker of tvm_stack_make_array for @p dtype: a zero of that type; bfloat16 is marked by
		 * a uint16 zero, as TVM's bfloat16 legalization would change a bfloat16 value (restored by
		 * restoreBF16ArrayTypes, pipeline.cpp; MDL tensors are never uint16).
		 */
		PrimExpr typeMarker(DataType dtype) { return zero(isBFloat16(dtype) ? DataType::UInt(16) : dtype); }

		/**
		 * @brief A buffer of @p dtype and @p shape at @p elemOffset (in elements) of the memory it is declared over (see
		 * tirx::DeclBuffer).
		 */
		tirx::BufferVar bufferOf(const std::string& name, DataType dtype, Array<PrimExpr> shape, PrimExpr elemOffset) {
			return tirx::BufferVar(
					name, tirx::BufferType("", dtype, std::move(shape), {}, std::move(elemOffset), 0, 0)
			);
		}

		Expr cpackedCall(const Array<Expr>& args) {
			return ::tvm::Call(DataType::Int(32), tirx::builtin::tvm_call_cpacked(), args);
		}

		/** @brief A DLTensor (stack-allocated by the caller) for @p data with the shape and element type of @p buffer. */
		Expr makeArray(const Expr& data, const tirx::BufferVar& buffer, const Array<PrimExpr>& shape) {
			Array<Expr> dims;
			for (const auto& d : shape)
				dims.push_back(::tvm::prim::cast(DataType::Int(64), d));
			return ::tvm::Call(
					::tvm::PointerType::VoidPointerTy(), tirx::builtin::tvm_stack_make_array(),
					{data,
					 ::tvm::Call(::tvm::PointerType(DataType::Int(64)), tirx::builtin::tvm_stack_make_shape(), dims),
					 IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), static_cast<int>(shape.size())),
					 typeMarker(buffer->dtype), IntImm(DataType::Int(64), 0)}
			);
		}

		/**
		 * @brief The kernels that entry functions call instead of accessing tensor memory themselves (copies,
		 * accumulations, constants, the dimensions in the context header): the memory may be on a device, and the entry
		 * functions run on the host. They are scheduled like the other kernels.
		 */
		struct HelperKernels {
			std::string prefix;
			std::map<std::string, tirx::PrimFunc> functions; ///< by symbol

			static std::string key(DataType dtype) {
				return std::to_string(dtype->dtype.code) + "_" + std::to_string(dtype->dtype.bits);
			}

			/** @brief The symbol of the kernel @p name, created by @p make if it does not exist yet. */
			std::string get(const std::string& name, const std::function<tirx::PrimFunc()>& make) {
				const std::string symbol = prefix + name;
				if (!functions.contains(symbol))
					functions.emplace(
							symbol, ::tvm::WithAttr(make(), ::tvm::attr::kGlobalSymbol, ::tvm::ffi::String(symbol))
					);
				return symbol;
			}

			static tirx::PrimFunc fromTE(const Array<::tvm::ffi::ObjectRef>& args) {
				static const auto createPrimFunc = ::tvm::ffi::Function::GetGlobalRequired("te.CreatePrimFunc");
				return createPrimFunc(args, nullptr).cast<tirx::PrimFunc>();
			}

			/** @brief `out[i] = src[i]` (or `src[i] + acc[i]`) over n elements. */
			std::string elementwise(DataType dtype, bool add) {
				return get((add ? "accumulate_" : "copy_") + key(dtype), [&] {
					PrimVar n("n", DataType::Int(64));
					auto src = ::tvm::te::placeholder({n}, dtype, "src");
					auto acc = ::tvm::te::placeholder({n}, dtype, "acc");
					auto out = ::tvm::te::compute(
							{n},
							[&](const Array<PrimVar>& i) {
								PrimExpr v = src(Array<PrimExpr>{i[0]});
								return add ? v + acc(Array<PrimExpr>{i[0]}) : v;
							},
							"out"
					);
					return add ? fromTE({src, acc, out}) : fromTE({src, out});
				});
			}

			/**
			 * @brief `out[i] = v_i` for k scalar parameters v_0..v_{k-1} (16-bit floats are passed as float32: the
			 * calling convention has no 16-bit scalars).
			 */
			std::string fill(DataType dtype, size_t k) {
				return get("fill" + std::to_string(k) + "_" + key(dtype), [&] {
					const bool narrowFloat =
							dtype.bits() < 32 && (dtype.MatchesCode(kDLFloat) || dtype.MatchesElementType(kDLBfloat, 16));
					const DataType param = narrowFloat ? DataType::Float(32) : dtype;
					Array<::tvm::ffi::ObjectRef> args;
					std::vector<PrimVar> values;
					for (size_t j = 0; j < k; ++j) {
						values.emplace_back("v" + std::to_string(j), param);
						args.push_back(values.back());
					}
					auto out = ::tvm::te::compute(
							{IntImm(DataType::Int(64), static_cast<int64_t>(k))},
							[&](const Array<PrimVar>& i) {
								PrimExpr v = values.back();
								for (size_t j = k - 1; j-- > 0;)
									v = ::tvm::if_then_else(
											i[0] == IntImm(DataType::Int(64), static_cast<int64_t>(j)), values[j], v
									);
								return ::tvm::prim::cast(dtype, v);
							},
							"out"
					);
					args.push_back(out);
					return fromTE(args);
				});
			}
		};

		/** @brief Builds the entry PrimFunc of one Relax function. */
		class EntryBuilder {
		public:
			EntryBuilder(
					const IRModule& mod, const std::unordered_map<const ::tvm::GlobalVarNode*, std::string>& symbols,
					HelperKernels& helpers, bool profile
			)
					: mod(mod), symbols(symbols), helpers(helpers), profile(profile) {}

			tirx::PrimFunc
			build(const relax::Function& func, const std::string& symbol, const EntrySignature* packing) {
				if (packing != nullptr && packing->context)
					beginContext(packing->dims);
				for (size_t p = 0; p < func->params.size(); ++p) {
					const auto& param = func->params[p];
					const auto* info = tensorInfo(relax::GetType(param));
					if (info == nullptr || !info->GetShape().defined())
						unsupported("parameter " + std::string(param->name) + " (not a tensor of known shape)");
					const bool packed = packing != nullptr && p < packing->params.size() && packing->params[p];
					values[param.get()] =
							Value::tensor(packed ? packedView(*info, param->name) : addBufferParam(param->name, *info));
				}
				const auto* seq = func->body.as<relax::SeqExprNode>();
				if (seq == nullptr)
					unsupported("function body (expected a SeqExpr)");

				// Remember all bindings and resolve the returned tensors, so that the allocations of returned tensors
				// become the output parameters (destination passing style).
				for (const auto& block : seq->blocks)
					for (const auto& binding : block->bindings)
						if (const auto* vb = binding.as<relax::VarBindingNode>())
							bindingOf[vb->var.get()] = vb->value;
				std::vector<Expr> returned;
				collectReturned(seq->body, returned);
				for (size_t i = 0; i < returned.size(); ++i) {
					const auto* info = tensorInfo(relax::GetType(returned[i]));
					if (info == nullptr || !info->GetShape().defined())
						unsupported("return value (not a tensor of known shape)");
					const bool packed = packing != nullptr && i < packing->outputs.size() && packing->outputs[i];
					const std::string name = "output" + std::to_string(i);
					outputs.push_back(packed ? packedView(*info, name) : addBufferParam(name, *info));
					accumulated.push_back(
							packing != nullptr && i < packing->accumulate.size() && packing->accumulate[i]
					);
					if (accumulated.back())
						continue; // computed into a temporary and then added to the output buffer
					// Follow aliases (`output = alloc`) to the allocation that can write into the output directly.
					const auto* var = returned[i].as<::tvm::VarNode>();
					while (var != nullptr) {
						auto it = bindingOf.find(var);
						const auto* alias = it == bindingOf.end() ? nullptr : it->second.as<::tvm::VarNode>();
						if (alias == nullptr)
							break;
						var = alias;
					}
					if (var != nullptr)
						outputOf.emplace(var, outputs.back());
				}

				if (packing != nullptr && packing->context)
					endContext(packing->sizeSymbol.empty() ? Role::Consumer : Role::Producer);

				for (const auto& block : seq->blocks)
					for (const auto& binding : block->bindings)
						translate(binding);

				// Copy returned values that are not computed in place into the output buffers.
				for (size_t i = 0; i < returned.size(); ++i) {
					Value v = evaluate(returned[i]);
					if (v.kind != Value::Kind::Tensor)
						unsupported("return value (not a tensor)");
					if (accumulated[i])
						accumulate(v.buffer, outputs[i]);
					else if (!v.buffer.same_as(outputs[i]))
						copy(v.buffer, outputs[i]);
				}

				const std::string bodySymbol = headerDims.empty() ? symbol : symbol + "_impl";
				auto attrs = ::tvm::DictAttrs(
						{{::tvm::attr::kGlobalSymbol, ::tvm::ffi::String(bodySymbol)},
						 {tirx::attr::kNoAlias, true},
						 {kAOTEntryAttr, true}}
				);
				auto body = tirx::PrimFunc(
						params,
						stmts.empty() ? tirx::Stmt(tirx::Evaluate(IntImm(DataType::Int(32), 0)))
									  : tirx::SeqStmt::Flatten(stmts),
						::tvm::VoidType(), attrs
				);
				functions.emplace_back(bodySymbol, body);
				if (!headerDims.empty())
					functions.emplace_back(symbol, wrapForHeaderDims(body, bodySymbol, symbol));
				return body;
			}

			/** @brief The PrimFuncs to add (the entry, plus its inner function if dimensions come from the header). */
			std::vector<std::pair<std::string, tirx::PrimFunc>> functions;

			/**
			 * @brief The exported entry of a consumer whose dimensions partly come from the context header: it takes the
			 * arguments of @p inner without those dimensions, reads them from the header, and calls @p inner.
			 */
			tirx::PrimFunc
			wrapForHeaderDims(const tirx::PrimFunc& inner, const std::string& innerSymbol, const std::string& symbol) {
				// Fresh variables for the outer function (its buffers have the same shapes as the inner ones).
				::tvm::ffi::Map<::tvm::Var, PrimExpr> rename;
				auto fresh = [&](const ::tvm::Var& v) -> PrimVar {
					if (auto it = rename.find(v); it != rename.end())
						return (*it).second.as_or_throw<PrimVar>();
					PrimVar copy(v->name, v->ty.as_or_throw<DataType>());
					rename.Set(v, copy);
					return copy;
				};
				Array<::tvm::Var> outerParams;
				Array<Expr> callArgs{StringImm(innerSymbol)};
				std::optional<tirx::BufferVar> outerContext;
				for (const auto& p : inner->params) {
					if (p->ty.as<tirx::BufferTypeNode>() == nullptr) { // a dimension from the header
						callArgs.push_back(fresh(p));
						continue;
					}
					const tirx::BufferVar buffer = tirx::GetBufferVar(p.get());
					for (const auto& d : buffer->shape)
						for (const auto& v : tirx::UndefinedVars(d))
							fresh(v);
					auto shape = substitute(buffer->shape, rename);
					auto outer = tirx::decl_buffer(shape, buffer->dtype, buffer.name());
					outerParams.push_back(outer);
					if (p.same_as(context))
						outerContext = outer;
					callArgs.push_back(makeArray(outer.data(), outer, shape));
				}
				// The header (the context's first contextDims words) is copied to the host stack: the context may be
				// device memory (on the CPU, the copy is a memcpy of 64 bytes).
				const auto n = IntImm(DataType::Int(64), kHeaderBytes / 8);
				auto deviceWords = bufferOf("ctx_header", DataType::Int(64), {n}, IntImm(DataType::Int(64), 0));
				auto words = bufferOf("ctx_words", DataType::Int(64), {n}, IntImm(DataType::Int(64), 0));
				std::vector<tirx::Stmt> body{
						tirx::DeclBuffer(deviceWords, outerContext->data()),
						tirx::DeclBuffer(
								words, ::tvm::Call(
											   ::tvm::PointerType(DataType::Int(64)), tirx::builtin::tvm_stack_alloca(),
											   {StringImm("shape"), n}
									   )
						),
				};
				body.push_back(tirx::Evaluate(::tvm::Call(
						DataType::Int(32), tirx::builtin::tvm_call_packed(),
						{StringImm("runtime.TVMTensorCopyToBytes"), makeArray(deviceWords.data(), deviceWords, {n}),
						 words.data(), IntImm(DataType::Int(64), kHeaderBytes)}
				)));
				for (const auto& [slot, dim] : headerDims) {
					PrimExpr value = tirx::BufferLoad(words, {IntImm(DataType::Int(64), slot)});
					if (dim.ty() != DataType::Int(64))
						value = ::tvm::prim::cast(dim.ty(), value);
					body.push_back(tirx::Bind(fresh(dim), value));
				}
				body.push_back(tirx::Evaluate(cpackedCall(callArgs)));
				auto attrs = ::tvm::DictAttrs(
						{{::tvm::attr::kGlobalSymbol, ::tvm::ffi::String(symbol)},
						 {tirx::attr::kNoAlias, true},
						 {kAOTEntryAttr, true}}
				);
				return tirx::PrimFunc(outerParams, tirx::SeqStmt::Flatten(body), ::tvm::VoidType(), attrs);
			}

			/** @brief The size of the context buffer in bytes (valid after build with packing). */
			PrimExpr contextBytes() const { return contextSize; }

		private:
			const IRModule& mod;
			const std::unordered_map<const ::tvm::GlobalVarNode*, std::string>& symbols;
			HelperKernels& helpers;
			bool profile;
			Array<::tvm::Var> params; ///< buffers and scalar dimensions
			std::vector<tirx::BufferVar> outputs;
			std::vector<bool> accumulated;
			std::vector<tirx::Stmt> stmts;
			std::unordered_map<const ::tvm::VarNode*, Value> values;
			std::unordered_map<const ::tvm::VarNode*, Expr> bindingOf;
			std::unordered_map<const ::tvm::VarNode*, tirx::BufferVar> outputOf;
			int counter = 0;
			tirx::BufferVar context; ///< the context buffer (uint8), if the function has packed tensors
			PrimExpr contextSize = IntImm(DataType::Int(64), 0); ///< bytes used so far

			static PrimExpr alignTo64(PrimExpr bytes) { return ::tvm::floordiv(bytes + 63, 64) * 64; }

			enum class Role { Producer, Consumer };
			static constexpr int64_t kHeaderBytes = 64; ///< the runtime dimensions as int64 (at most 8)
			Array<PrimVar> contextDims;
			tirx::BufferVar header;
			std::vector<const ::tvm::VarNode*> boundByParams; ///< dimension variables of the (non-packed) parameters

			void beginContext(const Array<PrimVar>& dims) {
				context = tirx::decl_buffer({PrimVar("ctx_bytes", DataType::Int(64))}, DataType::UInt(8), "ctx");
				contextDims = dims;
				// The context memory is only accessed as int64 words (one element type, as TIR's storage rewriting
				// requires): the header (the runtime dimensions) and the addresses of the packed tensors, which follow it.
				header = bufferOf(
						"ctx_words", DataType::Int(64), {::tvm::floordiv(context->shape[0], 8)},
						IntImm(DataType::Int(64), 0)
				);
				stmts.push_back(tirx::DeclBuffer(header, context.data()));
				contextSize = IntImm(DataType::Int(64), kHeaderBytes);
			}

			std::vector<tirx::Stmt> contextStmts; ///< the declarations of the packed tensors (after the prologue)
			std::vector<std::pair<int64_t, PrimVar>> headerDims; ///< (header slot, dimension) read by the outer entry

			/** @brief Places the next packed tensor in the context buffer (its own pointer to its 64-byte aligned offset). */
			tirx::BufferVar packedView(const relax::TensorTypeNode& info, const std::string& name) {
				const PrimExpr offset = alignTo64(contextSize);
				const DataType dtype = info.dtype.value();
				PrimExpr bytes = IntImm(DataType::Int(64), (dtype.bits() * dtype.lanes() + 7) / 8);
				for (const auto& d : info.GetShape().value())
					bytes = bytes * ::tvm::prim::cast(DataType::Int(64), d);
				contextSize = offset + bytes;
				auto buffer = bufferOf(name, dtype, info.GetShape().value(), IntImm(DataType::Int(64), 0));
				auto data = ::tvm::Call(
						::tvm::PointerType(dtype), tirx::builtin::address_of(),
						{tirx::BufferLoad(header, {::tvm::floordiv(offset, 8)})}
				);
				contextStmts.push_back(tirx::DeclBuffer(buffer, data));
				return buffer;
			}

			/**
			 * @brief Adds the context buffer as the last parameter, transfers the runtime dimensions through its header,
			 * and checks that it is large enough.
			 */
			void endContext(Role role) {
				contextSize = alignTo64(contextSize);
				params.push_back(context);
				std::vector<tirx::Stmt> prologue;
				if (role == Role::Producer && !contextDims.empty()) {
					// The dimensions into the header (through a kernel: the context may be device memory).
					auto dims = bufferOf(
							"ctx_dims", DataType::Int(64), {IntImm(DataType::Int(64), static_cast<int64_t>(contextDims.size()))},
							IntImm(DataType::Int(64), 0)
					);
					prologue.push_back(tirx::DeclBuffer(dims, context.data()));
					std::vector<PrimExpr> values;
					for (const auto& dim : contextDims)
						values.push_back(::tvm::prim::cast(DataType::Int(64), dim));
					std::vector<tirx::Stmt> saved = std::move(stmts);
					stmts.clear();
					fill(dims, values);
					prologue.insert(prologue.end(), stmts.begin(), stmts.end());
					stmts = std::move(saved);
				}
				for (size_t k = 0; k < contextDims.size(); ++k) {
					const PrimVar& dim = contextDims[k];
					if (role == Role::Producer) {
						continue;
					} else if (std::find(boundByParams.begin(), boundByParams.end(), dim.get()) ==
							   boundByParams.end()) {
						// Not determined by another argument (e.g. the sequence length when only [batch, classes]
						// cotangents are passed): a scalar parameter of this (inner) function, which the outer entry
						// function reads from the header (see wrapForHeaderDims). (A binding in the body would not
						// do: TIR passes hoist allocations and expressions that use it before it.)
						params.push_back(dim);
						headerDims.emplace_back(static_cast<int64_t>(k), dim);
					}
				}
				prologue.push_back(tirx::AssertStmt(
						context->shape[0] >= contextSize, StringImm("ValueError"),
						{StringImm("the context buffer is too small for these dimensions")}
				));
				// After the declaration of the context memory (the first statement), before everything that uses the
				// dimensions: the dimensions, the size check, then the packed tensors.
				prologue.insert(prologue.end(), contextStmts.begin(), contextStmts.end());
				stmts.insert(stmts.begin() + 1, prologue.begin(), prologue.end());
			}

			tirx::BufferVar addBufferParam(const std::string& name, const relax::TensorTypeNode& info) {
				auto buffer = tirx::decl_buffer(info.GetShape().value(), info.dtype.value(), name);
				for (const auto& d : info.GetShape().value())
					if (const auto* v = d.as<::tvm::VarNode>())
						boundByParams.push_back(v);
				params.push_back(buffer);
				return buffer;
			}

			void collectReturned(const Expr& e, std::vector<Expr>& out) {
				if (const auto* tuple = e.as<relax::TupleNode>()) {
					for (const auto& field : tuple->fields)
						collectReturned(field, out);
					return;
				}
				if (const auto* var = e.as<::tvm::VarNode>()) {
					if (auto it = bindingOf.find(var); it != bindingOf.end() && it->second.as<relax::TupleNode>()) {
						collectReturned(it->second, out);
						return;
					}
				}
				out.push_back(e);
			}

			std::string freshName(const std::string& base) { return base + "_" + std::to_string(counter++); }

			Value evaluate(const Expr& e) {
				if (const auto* var = e.as<::tvm::VarNode>()) {
					auto it = values.find(var);
					if (it != values.end())
						return it->second;
					if (auto prim = scalarValue(e)) // a symbolic dimension (e.g. of the parameters' shapes)
						return Value::primitive(*prim);
					unsupported("use of undefined variable " + std::string(var->name));
				}
				if (auto prim = scalarValue(e))
					return Value::primitive(*prim);
				if (const auto* se = e.as<relax::ShapeExprNode>())
					return Value::shapeOf(se->values);
				if (const auto* tuple = e.as<relax::TupleNode>()) {
					Value v{.kind = Value::Kind::Tuple};
					for (const auto& field : tuple->fields)
						v.fields.push_back(evaluate(field));
					return v;
				}
				if (const auto* get = e.as<relax::TupleGetItemNode>()) {
					Value tuple = evaluate(get->tuple);
					if (tuple.kind != Value::Kind::Tuple || get->index >= static_cast<int>(tuple.fields.size()))
						unsupported("tuple access");
					return tuple.fields[get->index];
				}
				if (auto c = tensorConstant(e))
					return Value::tensor(materializeConstant(*c));
				unsupported("expression " + std::string(e->GetTypeKey()));
			}

			/** @brief Small constants (e.g. scalars that were not inlined) are written into a workspace buffer. */
			tirx::BufferVar materializeConstant(const ::tvm::runtime::Tensor& data) {
				const int64_t n = data.Shape().Product();
				const DataType dtype(data->dtype);
				const bool f32 = dtype == DataType::Float(32), i32 = dtype == DataType::Int(32);
				const bool f16 = dtype == DataType::Float(16), bf16 = dtype == DataType::BFloat(16);
				if (n > 1024 || !(f32 || i32 || f16 || bf16))
					unsupported("constant (only float and int32 constants with up to 1024 elements are supported)");
				auto element = [&](int64_t i) -> PrimExpr {
					if (i32)
						return IntImm(dtype, static_cast<const int32_t*>(data->data)[i]);
					if (f32)
						return ::tvm::FloatImm(dtype, static_cast<const float*>(data->data)[i]);
					const uint16_t bits = static_cast<const uint16_t*>(data->data)[i];
					if (f16) {
						_Float16 half;
						std::memcpy(&half, &bits, sizeof(half));
						return ::tvm::FloatImm(dtype, static_cast<double>(half));
					}
					const uint32_t wide = static_cast<uint32_t>(bits) << 16; // bfloat16: the upper half of a float32
					float value;
					std::memcpy(&value, &wide, sizeof(value));
					return ::tvm::FloatImm(dtype, value);
				};
				Array<PrimExpr> shape;
				for (int64_t d : data.Shape())
					shape.push_back(IntImm(DataType::Int(64), d));
				auto buffer = tirx::decl_buffer(shape, dtype, freshName("const"), "global.workspace");
				stmts.push_back(tirx::AllocBuffer(buffer));
				auto flat = bufferOf(
						freshName("const_flat"), dtype, {IntImm(DataType::Int(64), n)}, IntImm(DataType::Int(64), 0)
				);
				stmts.push_back(tirx::DeclBuffer(flat, buffer.data()));
				std::vector<PrimExpr> values;
				for (int64_t i = 0; i < n; ++i)
					values.push_back(element(i));
				fill(flat, values);
				return buffer;
			}

			tirx::BufferVar
			view(const tirx::BufferVar& base, Array<PrimExpr> shape, DataType dtype, PrimExpr elemOffset) {
				auto buffer = bufferOf(freshName("view"), dtype, std::move(shape), std::move(elemOffset));
				stmts.push_back(tirx::DeclBuffer(buffer, base.data()));
				return buffer;
			}

			tirx::BufferVar allocate(Array<PrimExpr> shape, DataType dtype, const std::string& name) {
				// Workspace scope: allocated with TVMBackendAllocWorkspace even if small. (Small "global" allocations
				// become stack allocations, which are not aligned as the kernels assume.)
				auto buffer = tirx::decl_buffer(std::move(shape), dtype, freshName(name), "global.workspace");
				stmts.push_back(tirx::AllocBuffer(buffer));
				return buffer;
			}

			/** @brief The number of elements of @p buffer. */
			static PrimExpr numElements(const tirx::BufferVar& buffer) {
				PrimExpr n = IntImm(DataType::Int(64), 1);
				for (const auto& d : buffer->shape)
					n = n * ::tvm::prim::cast(DataType::Int(64), d);
				return n;
			}

			/** @brief Calls a helper kernel (see HelperKernels) with tensor and scalar arguments. */
			void callHelper(const std::string& symbol, const std::vector<Value>& args) {
				Array<Expr> callArgs{StringImm(symbol)};
				for (const auto& arg : args)
					callArgs.push_back(packedArg(arg));
				stmts.push_back(tirx::Evaluate(cpackedCall(callArgs)));
			}

			/** @brief Writes @p values into the 1-D buffer @p dst (through a kernel: dst may be device memory). */
			void fill(const tirx::BufferVar& dst, const std::vector<PrimExpr>& values) {
				const DataType dtype = dst->dtype;
				const bool narrowFloat =
						dtype.bits() < 32 && (dtype.MatchesCode(kDLFloat) || dtype.MatchesElementType(kDLBfloat, 16));
				std::vector<Value> args;
				for (const auto& v : values)
					args.push_back(Value::primitive(::tvm::prim::cast(narrowFloat ? DataType::Float(32) : dtype, v)));
				args.push_back(Value::tensor(dst));
				callHelper(helpers.fill(dtype, values.size()), args);
			}

			void copy(const tirx::BufferVar& src, const tirx::BufferVar& dst) {
				PrimExpr n = numElements(dst);
				auto flatSrc = view(src, {n}, src->dtype, src->elem_offset);
				auto flatDst = view(dst, {n}, dst->dtype, dst->elem_offset);
				callHelper(helpers.elementwise(dst->dtype, false), {Value::tensor(flatSrc), Value::tensor(flatDst)});
			}

			/** @brief dst += src (elementwise, same shapes). */
			void accumulate(const tirx::BufferVar& src, const tirx::BufferVar& dst) {
				PrimExpr n = numElements(dst);
				auto flatSrc = view(src, {n}, src->dtype, src->elem_offset);
				auto flatDst = view(dst, {n}, dst->dtype, dst->elem_offset);
				callHelper(
						helpers.elementwise(dst->dtype, true),
						{Value::tensor(flatSrc), Value::tensor(flatDst), Value::tensor(flatDst)}
				);
			}

			Expr packedArg(const Value& v) {
				switch (v.kind) {
				case Value::Kind::Prim:
					return v.prim;
				case Value::Kind::Tensor: {
					// Kernels expect byte_offset 0: a view at an offset (e.g. into the context buffer) is passed as a
					// pointer to its first element.
					Expr data = v.buffer.data();
					if (!::tvm::prim::is_zero(v.buffer->elem_offset)) {
						Array<PrimExpr> origin;
						for (const auto& d : v.buffer->shape)
							origin.push_back(zero(d.ty()));
						data = ::tvm::Call(
								::tvm::PointerType(v.buffer->dtype), tirx::builtin::address_of(),
								{tirx::BufferLoad(v.buffer, origin)}
						);
					}
					return makeArray(data, v.buffer, v.buffer->shape);
				}
				default:
					unsupported("kernel argument");
				}
			}

			void callKernel(const GlobalVar& gv, const std::vector<Value>& args) {
				auto it = symbols.find(gv.get());
				if (it == symbols.end())
					unsupported("call of " + std::string(gv->name_hint) + " (not a PrimFunc)");
				Array<Expr> callArgs{StringImm(it->second)};
				for (const auto& arg : args) {
					if (arg.kind == Value::Kind::Shape) { // unpacked symbolic variables (call_tir_dyn)
						for (const auto& d : arg.shape)
							callArgs.push_back(::tvm::prim::cast(DataType::Int(64), d));
					} else {
						callArgs.push_back(packedArg(arg));
					}
				}
				auto hook = [&](const char* fn) {
					return tirx::Evaluate(::tvm::Call(
							DataType::Int(32), tirx::builtin::call_extern(), {StringImm(fn), StringImm(it->second)}
					));
				};
				if (profile)
					stmts.push_back(hook("mdl_profile_begin"));
				stmts.push_back(tirx::Evaluate(cpackedCall(callArgs)));
				if (profile)
					stmts.push_back(hook("mdl_profile_end"));
			}

			static DataType dtypeArg(const Expr& e) {
				const auto* imm = e.as<::tvm::GenericConstNode>();
				auto dtype = imm == nullptr ? std::nullopt : imm->value.try_cast<DLDataType>();
				if (!dtype)
					unsupported("dtype argument");
				return DataType(*dtype);
			}

			Array<PrimExpr> shapeArg(const Expr& e) {
				Value v = evaluate(e);
				if (v.kind != Value::Kind::Shape)
					unsupported("shape argument");
				return v.shape;
			}

			static int64_t bytesOf(DataType dtype) { return (dtype.bits() * dtype.lanes() + 7) / 8; }

			/**
			 * @brief `var = if cond: a else: b` (a runtime guard): both branches compute into the same buffer (the
			 * output, if @p var is returned); a branch's allocations are local to it.
			 */
			void translateIf(const ::tvm::VarNode* var, const relax::IfNode& branch) {
				Value cond = evaluate(branch.cond);
				if (cond.kind != Value::Kind::Prim)
					unsupported("condition (not a scalar)");
				const auto* info = tensorInfo(relax::GetType(::tvm::ffi::GetRef<::tvm::Var>(var)));
				if (info == nullptr || !info->GetShape().defined())
					unsupported("if (the result is not a tensor of known shape)");
				tirx::BufferVar result;
				if (auto out = outputOf.find(var); out != outputOf.end())
					result = out->second;
				else
					result = allocate(info->GetShape().value(), info->dtype.value(), "result");
				auto lower = [&](const Expr& e) {
					const auto* seq = e.as<relax::SeqExprNode>();
					if (seq == nullptr)
						unsupported("if branch (expected a SeqExpr)");
					std::vector<tirx::Stmt> outer = std::move(stmts);
					stmts.clear();
					for (const auto& block : seq->blocks)
						for (const auto& b : block->bindings)
							if (const auto* vb = b.as<relax::VarBindingNode>())
								bindingOf[vb->var.get()] = vb->value;
					// The branch's result: computed into the result buffer directly if it is an allocation.
					const auto* resultVar = seq->body.as<::tvm::VarNode>();
					while (resultVar != nullptr) {
						auto it = bindingOf.find(resultVar);
						const auto* alias = it == bindingOf.end() ? nullptr : it->second.as<::tvm::VarNode>();
						if (alias == nullptr)
							break;
						resultVar = alias;
					}
					if (resultVar != nullptr)
						outputOf.emplace(resultVar, result);
					for (const auto& block : seq->blocks)
						for (const auto& b : block->bindings)
							translate(b);
					Value v = evaluate(seq->body);
					if (v.kind != Value::Kind::Tensor)
						unsupported("if branch result (not a tensor)");
					if (!v.buffer.same_as(result))
						copy(v.buffer, result);
					std::vector<tirx::Stmt> inner = std::move(stmts);
					stmts = std::move(outer);
					return inner.empty() ? tirx::Stmt(tirx::Evaluate(IntImm(DataType::Int(32), 0)))
										 : tirx::SeqStmt::Flatten(inner);
				};
				auto thenStmt = lower(branch.true_branch);
				auto elseStmt = lower(branch.false_branch);
				stmts.push_back(tirx::IfThenElse(cond.prim, thenStmt, elseStmt));
				values[var] = Value::tensor(result);
			}

			void translate(const relax::Binding& binding) {
				const ::tvm::VarNode* var = binding->var.get();
				if (const auto* mc = binding.as<relax::MatchCastNode>()) {
					values[var] = evaluate(mc->value);
					return;
				}
				const Expr& value = binding.as<relax::VarBindingNode>()->value;
				if (const auto* branch = value.as<relax::IfNode>()) {
					translateIf(var, *branch);
					return;
				}
				const auto* call = value.as<::tvm::CallNode>();
				if (call == nullptr) {
					values[var] = evaluate(value);
					return;
				}

				static const auto& builtinAlloc = ::tvm::Op::Get("relax.builtin.alloc_tensor");
				static const auto& allocStorage = ::tvm::Op::Get("relax.memory.alloc_storage");
				static const auto& allocTensor = ::tvm::Op::Get("relax.memory.alloc_tensor");
				static const auto& killStorage = ::tvm::Op::Get("relax.memory.kill_storage");
				static const auto& killTensor = ::tvm::Op::Get("relax.memory.kill_tensor");
				static const auto& reshape = ::tvm::Op::Get("relax.reshape");
				static const auto& callTirDyn = ::tvm::Op::Get("relax.vm.call_tir_dyn");

				if (const auto* gv = call->op.as<::tvm::GlobalVarNode>()) {
					std::vector<Value> args;
					for (const auto& arg : call->args)
						args.push_back(evaluate(arg));
					callKernel(::tvm::ffi::GetRef<GlobalVar>(gv), args);
				} else if (call->op.same_as(callTirDyn)) {
					std::vector<Value> args;
					for (const auto& arg : call->args[1].as_or_throw<relax::Tuple>()->fields)
						args.push_back(evaluate(arg));
					callKernel(call->args[0].as_or_throw<GlobalVar>(), args);
				} else if (call->op.same_as(builtinAlloc)) {
					if (auto out = outputOf.find(var); out != outputOf.end())
						values[var] = Value::tensor(out->second); // returned tensor: compute into the output buffer
					else
						values[var] =
								Value::tensor(allocate(shapeArg(call->args[0]), dtypeArg(call->args[1]), "tensor"));
				} else if (call->op.same_as(allocStorage)) {
					// Storage of `size` bytes; allocated as elements of the storage dtype.
					Array<PrimExpr> size = shapeArg(call->args[0]);
					DataType dtype = dtypeArg(call->args[3]);
					PrimExpr bytes = ::tvm::prim::cast(DataType::Int(64), size[0]);
					PrimExpr elems = ::tvm::floordiv(
							bytes + IntImm(DataType::Int(64), bytesOf(dtype) - 1),
							IntImm(DataType::Int(64), bytesOf(dtype))
					);
					values[var] = Value::tensor(allocate({elems}, dtype, "storage"));
				} else if (call->op.same_as(allocTensor)) {
					Value storage = evaluate(call->args[0]);
					DataType dtype = dtypeArg(call->args[3]);
					if (storage.kind != Value::Kind::Tensor || storage.buffer->dtype != dtype)
						unsupported("alloc_tensor on storage of another dtype");
					PrimExpr offset = ::tvm::floordiv(
							::tvm::prim::cast(DataType::Int(64), evaluate(call->args[1]).prim),
							IntImm(DataType::Int(64), bytesOf(dtype))
					);
					values[var] = Value::tensor(view(storage.buffer, shapeArg(call->args[2]), dtype, offset));
				} else if (call->op.same_as(reshape)) {
					Value x = evaluate(call->args[0]);
					if (x.kind != Value::Kind::Tensor)
						unsupported("reshape of a non-tensor");
					values[var] = Value::tensor(
							view(x.buffer, shapeArg(call->args[1]), x.buffer->dtype, x.buffer->elem_offset)
					);
				} else if (call->op.same_as(killStorage) || call->op.same_as(killTensor)) {
					// Lifetimes are handled by the workspace allocations.
				} else {
					unsupported("operation " + std::string(call->op.as_or_throw<::tvm::Op>()->name));
				}
			}
		};
	} // namespace

	namespace {
		/** @brief `size(dims...) -> bytes`: the size of a context buffer for the given dimensions of `...`. */
		tirx::PrimFunc
		contextSizeFunction(const PrimExpr& bytes, const Array<PrimVar>& dims, const std::string& symbol) {
			for (const auto& var : tirx::UndefinedVars(bytes))
				if (std::find_if(dims.begin(), dims.end(), [&](const PrimVar& d) { return d.same_as(var); }) ==
					dims.end())
					unsupported(
							"context buffer layout (its size depends on " + std::string(var->name) +
							", which is not a dimension of `...`)"
					);
			auto attrs =
					::tvm::DictAttrs({{::tvm::attr::kGlobalSymbol, ::tvm::ffi::String(symbol)}, {kAOTEntryAttr, true}});
			Array<::tvm::Var> params;
			for (const auto& d : dims)
				params.push_back(d);
			return tirx::PrimFunc(params, tirx::Return(bytes), DataType::Int(64), attrs);
		}
	} // namespace

	IRModule lowerToAOT(
			IRModule mod, const std::string& kernelPrefix,
			const std::unordered_map<std::string, EntrySignature>& packing, bool profile
	) {
		// Give every kernel a unique global symbol.
		std::unordered_map<const ::tvm::GlobalVarNode*, std::string> symbols;
		IRModule result =
				IRModule(::tvm::ffi::Map<GlobalVar, ::tvm::BaseFunc>(), mod->source_map, mod->attrs, mod->global_infos);
		for (const auto& [gv, func] : mod->functions) {
			if (const auto* prim = func.as<tirx::PrimFuncNode>()) {
				std::string symbol = kernelPrefix + std::string(gv->name_hint);
				symbols[gv.get()] = symbol;
				result->Add(
						gv, ::tvm::WithAttr(
									::tvm::ffi::GetRef<tirx::PrimFunc>(prim), ::tvm::attr::kGlobalSymbol,
									::tvm::ffi::String(symbol)
							)
				);
			}
		}
		HelperKernels helpers{.prefix = kernelPrefix + "mdl_", .functions = {}};
		for (const auto& [gv, func] : mod->functions) {
			if (const auto* fn = func.as<relax::FunctionNode>()) {
				auto symbol = fn->GetAttr<::tvm::ffi::String>(::tvm::attr::kGlobalSymbol);
				if (!symbol)
					continue; // internal helper function
				EntryBuilder builder{mod, symbols, helpers, profile};
				auto packed = packing.find(std::string(symbol.value()));
				const EntrySignature* signature = packed == packing.end() ? nullptr : &packed->second;
				builder.build(::tvm::ffi::GetRef<relax::Function>(fn), symbol.value(), signature);
				for (const auto& [name, func] : builder.functions)
					result->Add(GlobalVar(name), func);
				if (signature != nullptr && !signature->sizeSymbol.empty())
					result->Add(
							GlobalVar(signature->sizeSymbol),
							contextSizeFunction(builder.contextBytes(), signature->dims, signature->sizeSymbol)
					);
			}
		}
		for (const auto& [symbol, func] : helpers.functions)
			result->Add(GlobalVar(symbol), func);
		return result;
	}

} // namespace mdl::backend::tvm
