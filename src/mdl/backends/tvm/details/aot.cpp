#include "./aot.hpp"

#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mdl::backend::tvm {

	using ::tvm::GlobalVar;
	using ::tvm::IntImm;
	using ::tvm::IRModule;
	using ::tvm::PrimExpr;
	using ::tvm::ffi::Array;
	namespace relax = ::tvm::relax;
	namespace tirx = ::tvm::tirx;
	using DataType = ::tvm::DataType;

	namespace {
		/** @brief The TIR representation of a Relax value. */
		struct Value {
			enum class Kind { None, Tensor, Prim, Shape, Tuple } kind = Kind::None;
			tirx::Buffer buffer;
			PrimExpr prim;
			Array<PrimExpr> shape;
			std::vector<Value> fields;

			static Value tensor(tirx::Buffer b) { return Value{.kind = Kind::Tensor, .buffer = std::move(b)}; }
			static Value primitive(PrimExpr p) { return Value{.kind = Kind::Prim, .prim = std::move(p)}; }
			static Value shapeOf(Array<PrimExpr> s) { return Value{.kind = Kind::Shape, .shape = std::move(s)}; }
		};

		[[noreturn]] void unsupported(const std::string& what) {
			throw std::runtime_error("mdl AOT lowering: unsupported " + what);
		}

		const relax::TensorStructInfoNode* tensorInfo(const relax::StructInfo& sinfo) {
			return sinfo.as<relax::TensorStructInfoNode>();
		}

		/** @brief Builds the entry PrimFunc of one Relax function. */
		class EntryBuilder {
		public:
			EntryBuilder(
					const IRModule& mod, const std::unordered_map<const ::tvm::GlobalVarNode*, std::string>& symbols,
					bool profile
			)
					: mod(mod), symbols(symbols), profile(profile) {}

			tirx::PrimFunc build(const relax::Function& func, const std::string& symbol, const EntrySignature* packing) {
				if (packing != nullptr && packing->context)
					beginContext();
				for (size_t p = 0; p < func->params.size(); ++p) {
					const auto& param = func->params[p];
					const auto* info = tensorInfo(relax::GetStructInfo(param));
					if (info == nullptr || !info->GetShape().defined())
						unsupported("parameter " + std::string(param->name_hint()) + " (not a tensor of known shape)");
					const bool packed = packing != nullptr && p < packing->params.size() && packing->params[p];
					values[param.get()] = Value::tensor(
							packed ? packedView(*info, param->name_hint()) : addBufferParam(param->name_hint(), *info)
					);
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
				std::vector<relax::Expr> returned;
				collectReturned(seq->body, returned);
				for (size_t i = 0; i < returned.size(); ++i) {
					const auto* info = tensorInfo(relax::GetStructInfo(returned[i]));
					if (info == nullptr || !info->GetShape().defined())
						unsupported("return value (not a tensor of known shape)");
					const bool packed = packing != nullptr && i < packing->outputs.size() && packing->outputs[i];
					const std::string name = "output" + std::to_string(i);
					outputs.push_back(packed ? packedView(*info, name) : addBufferParam(name, *info));
					accumulated.push_back(packing != nullptr && i < packing->accumulate.size() && packing->accumulate[i]);
					if (accumulated.back())
						continue; // computed into a temporary and then added to the output buffer
					// Follow aliases (`output = alloc`) to the allocation that can write into the output directly.
					const auto* var = returned[i].as<relax::VarNode>();
					while (var != nullptr) {
						auto it = bindingOf.find(var);
						const auto* alias = it == bindingOf.end() ? nullptr : it->second.as<relax::VarNode>();
						if (alias == nullptr)
							break;
						var = alias;
					}
					if (var != nullptr)
						outputOf.emplace(var, outputs.back());
				}

				if (packing != nullptr && packing->context)
					endContext();

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

				auto attrs = ::tvm::DictAttrs(
						{{::tvm::attr::kGlobalSymbol, ::tvm::ffi::String(symbol)},
						 {tirx::attr::kNoAlias, ::tvm::Bool(true)},
						 {kAOTEntryAttr, ::tvm::Bool(true)}}
				);
				return tirx::PrimFunc(
						params, stmts.empty() ? tirx::Stmt(tirx::Evaluate(0)) : tirx::SeqStmt::Flatten(stmts),
						::tvm::VoidType(), bufferMap,
						attrs
				);
			}

			/** @brief The size of the context buffer in bytes (valid after build with packing). */
			PrimExpr contextBytes() const { return contextSize; }

		private:
			const IRModule& mod;
			const std::unordered_map<const ::tvm::GlobalVarNode*, std::string>& symbols;
			bool profile;
			Array<tirx::Var> params;
			::tvm::ffi::Map<tirx::Var, tirx::Buffer> bufferMap;
			std::vector<tirx::Buffer> outputs;
			std::vector<bool> accumulated;
			std::vector<tirx::Stmt> stmts;
			std::unordered_map<const relax::VarNode*, Value> values;
			std::unordered_map<const relax::VarNode*, relax::Expr> bindingOf;
			std::unordered_map<const relax::VarNode*, tirx::Buffer> outputOf;
			int counter = 0;
			tirx::Buffer context; ///< the context buffer (uint8), if the function has packed tensors
			tirx::Var contextHandle;
			PrimExpr contextSize = IntImm(DataType::Int(64), 0); ///< bytes used so far

			static PrimExpr alignTo64(PrimExpr bytes) { return ::tvm::floordiv(bytes + 63, 64) * 64; }

			void beginContext() {
				contextHandle = tirx::Var("ctx_handle", DataType::Handle());
				context = tirx::decl_buffer({tirx::Var("ctx_bytes", DataType::Int(64))}, DataType::UInt(8), "ctx");
			}

			/** @brief Places the next packed tensor in the context buffer. */
			tirx::Buffer packedView(const relax::TensorStructInfoNode& info, const std::string& name) {
				const PrimExpr offset = alignTo64(contextSize);
				PrimExpr bytes = IntImm(DataType::Int(64), info.dtype.bytes());
				for (const auto& d : info.GetShape().value())
					bytes = bytes * ::tvm::cast(DataType::Int(64), d);
				contextSize = offset + bytes;
				auto buffer = tirx::Buffer(
						context->data, info.dtype, info.GetShape().value(), {},
						::tvm::floordiv(offset, info.dtype.bytes()), name, 0, 0, tirx::BufferType::kDefault
				);
				stmts.push_back(tirx::DeclBuffer(buffer));
				return buffer;
			}

			/** @brief Adds the context buffer as the last parameter and checks that it is large enough. */
			void endContext() {
				contextSize = alignTo64(contextSize);
				params.push_back(contextHandle);
				bufferMap.Set(contextHandle, context);
				stmts.insert(
						stmts.begin(),
						tirx::AssertStmt(
								context->shape[0] >= contextSize, tirx::StringImm("ValueError"),
								{tirx::StringImm("the context buffer is too small for these dimensions")}
						)
				);
			}

			tirx::Buffer addBufferParam(const std::string& name, const relax::TensorStructInfoNode& info) {
				auto handle = tirx::Var(name + "_handle", DataType::Handle());
				auto buffer = tirx::decl_buffer(info.GetShape().value(), info.dtype, name);
				params.push_back(handle);
				bufferMap.Set(handle, buffer);
				return buffer;
			}

			void collectReturned(const relax::Expr& e, std::vector<relax::Expr>& out) {
				if (const auto* tuple = e.as<relax::TupleNode>()) {
					for (const auto& field : tuple->fields)
						collectReturned(field, out);
					return;
				}
				if (const auto* var = e.as<relax::VarNode>()) {
					if (auto it = bindingOf.find(var); it != bindingOf.end() && it->second.as<relax::TupleNode>()) {
						collectReturned(it->second, out);
						return;
					}
				}
				out.push_back(e);
			}

			std::string freshName(const std::string& base) { return base + "_" + std::to_string(counter++); }

			Value evaluate(const relax::Expr& e) {
				if (const auto* var = e.as<relax::VarNode>()) {
					auto it = values.find(var);
					if (it == values.end())
						unsupported("use of undefined variable " + std::string(var->name_hint()));
					return it->second;
				}
				if (const auto* pv = e.as<relax::PrimValueNode>())
					return Value::primitive(pv->value);
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
				if (const auto* c = e.as<relax::ConstantNode>())
					return Value::tensor(materializeConstant(*c));
				unsupported("expression " + std::string(e->GetTypeKey()));
			}

			/** @brief Small constants (e.g. scalars that were not inlined) are written into a stack buffer. */
			tirx::Buffer materializeConstant(const relax::ConstantNode& c) {
				const auto& data = c.data;
				const int64_t n = data.Shape().Product();
				if (n > 1024 || data->dtype.code != kDLFloat || data->dtype.bits != 32)
					unsupported("constant (only float32 constants with up to 1024 elements are supported)");
				Array<PrimExpr> shape;
				for (int64_t d : data.Shape())
					shape.push_back(IntImm(DataType::Int(64), d));
				auto buffer = tirx::decl_buffer(shape, DataType::Float(32), freshName("const"), "global.workspace");
				stmts.push_back(tirx::AllocBuffer(buffer));
				auto flat = tirx::decl_buffer({IntImm(DataType::Int(64), n)}, DataType::Float(32), freshName("const_flat"));
				flat.CopyOnWrite()->data = buffer->data;
				stmts.push_back(tirx::DeclBuffer(flat));
				const auto* values = static_cast<const float*>(data->data);
				for (int64_t i = 0; i < n; ++i)
					stmts.push_back(tirx::BufferStore(
							flat, ::tvm::FloatImm(DataType::Float(32), values[i]), {IntImm(DataType::Int(64), i)}
					));
				return buffer;
			}

			tirx::Buffer view(const tirx::Buffer& base, Array<PrimExpr> shape, DataType dtype, PrimExpr elemOffset) {
				auto buffer = tirx::Buffer(
						base->data, dtype, std::move(shape), {}, std::move(elemOffset), freshName("view"), 0, 0,
						tirx::BufferType::kDefault
				);
				stmts.push_back(tirx::DeclBuffer(buffer));
				return buffer;
			}

			tirx::Buffer allocate(Array<PrimExpr> shape, DataType dtype, const std::string& name) {
				// Workspace scope: allocated with TVMBackendAllocWorkspace even if small. (Small "global" allocations
				// become stack allocations, which are not aligned as the kernels assume.)
				auto buffer = tirx::decl_buffer(std::move(shape), dtype, freshName(name), "global.workspace");
				stmts.push_back(tirx::AllocBuffer(buffer));
				return buffer;
			}

			void copy(const tirx::Buffer& src, const tirx::Buffer& dst) {
				PrimExpr n = IntImm(DataType::Int(64), 1);
				for (const auto& d : dst->shape)
					n = n * ::tvm::cast(DataType::Int(64), d);
				auto flatSrc = view(src, {n}, src->dtype, src->elem_offset);
				auto flatDst = view(dst, {n}, dst->dtype, dst->elem_offset);
				auto i = tirx::Var("i", DataType::Int(64));
				stmts.push_back(tirx::For(
						i, IntImm(DataType::Int(64), 0), n, tirx::ForKind::kSerial,
						tirx::BufferStore(flatDst, tirx::BufferLoad(flatSrc, {i}), {i})
				));
			}

			/** @brief dst += src (elementwise, same shapes). */
			void accumulate(const tirx::Buffer& src, const tirx::Buffer& dst) {
				PrimExpr n = IntImm(DataType::Int(64), 1);
				for (const auto& d : dst->shape)
					n = n * ::tvm::cast(DataType::Int(64), d);
				auto flatSrc = view(src, {n}, src->dtype, src->elem_offset);
				auto flatDst = view(dst, {n}, dst->dtype, dst->elem_offset);
				auto i = tirx::Var("i", DataType::Int(64));
				stmts.push_back(tirx::For(
						i, IntImm(DataType::Int(64), 0), n, tirx::ForKind::kSerial,
						tirx::BufferStore(flatDst, tirx::BufferLoad(flatDst, {i}) + tirx::BufferLoad(flatSrc, {i}), {i})
				));
			}

			PrimExpr packedArg(const Value& v) {
				switch (v.kind) {
				case Value::Kind::Prim:
					return v.prim;
				case Value::Kind::Tensor: {
					Array<PrimExpr> dims;
					for (const auto& d : v.buffer->shape)
						dims.push_back(::tvm::cast(DataType::Int(64), d));
					auto shape = tirx::Call(DataType::Handle(), tirx::builtin::tvm_stack_make_shape(), dims);
					// Kernels expect byte_offset 0: a view at an offset (e.g. into the context buffer) is passed as a
					// pointer to its first element.
					PrimExpr data = v.buffer->data;
					if (!tirx::is_zero(v.buffer->elem_offset)) {
						Array<PrimExpr> origin;
						for (const auto& d : v.buffer->shape)
							origin.push_back(tirx::make_zero(d.dtype()));
						data = tirx::Call(
								DataType::Handle(), tirx::builtin::address_of(), {tirx::BufferLoad(v.buffer, origin)}
						);
					}
					return tirx::Call(
							DataType::Handle(), tirx::builtin::tvm_stack_make_array(),
							{data, shape, IntImm(DataType::Int(32), 0),
							 IntImm(DataType::Int(32), static_cast<int>(v.buffer->shape.size())),
							 tirx::make_zero(v.buffer->dtype), IntImm(DataType::Int(64), 0)}
					);
				}
				default:
					unsupported("kernel argument");
				}
			}

			void callKernel(const GlobalVar& gv, const std::vector<Value>& args) {
				auto it = symbols.find(gv.get());
				if (it == symbols.end())
					unsupported("call of " + std::string(gv->name_hint) + " (not a PrimFunc)");
				Array<PrimExpr> callArgs{tirx::StringImm(it->second)};
				for (const auto& arg : args) {
					if (arg.kind == Value::Kind::Shape) { // unpacked symbolic variables (call_tir_dyn)
						for (const auto& d : arg.shape)
							callArgs.push_back(::tvm::cast(DataType::Int(64), d));
					} else {
						callArgs.push_back(packedArg(arg));
					}
				}
				auto hook = [&](const char* fn) {
					return tirx::Evaluate(tirx::Call(
							DataType::Int(32), tirx::builtin::call_extern(), {tirx::StringImm(fn), tirx::StringImm(it->second)}
					));
				};
				if (profile)
					stmts.push_back(hook("mdl_profile_begin"));
				stmts.push_back(
						tirx::Evaluate(tirx::Call(DataType::Int(32), tirx::builtin::tvm_call_cpacked(), callArgs))
				);
				if (profile)
					stmts.push_back(hook("mdl_profile_end"));
			}

			static DataType dtypeArg(const relax::Expr& e) {
				const auto* imm = e.as<relax::DataTypeImmNode>();
				if (imm == nullptr)
					unsupported("dtype argument");
				return imm->value;
			}

			Array<PrimExpr> shapeArg(const relax::Expr& e) {
				Value v = evaluate(e);
				if (v.kind != Value::Kind::Shape)
					unsupported("shape argument");
				return v.shape;
			}

			void translate(const relax::Binding& binding) {
				const relax::VarNode* var = binding->var.get();
				if (const auto* mc = binding.as<relax::MatchCastNode>()) {
					values[var] = evaluate(mc->value);
					return;
				}
				const relax::Expr& value = binding.as<relax::VarBindingNode>()->value;
				const auto* call = value.as<relax::CallNode>();
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
					for (const auto& arg : ::tvm::Downcast<relax::Tuple>(call->args[1])->fields)
						args.push_back(evaluate(arg));
					callKernel(::tvm::Downcast<GlobalVar>(call->args[0]), args);
				} else if (call->op.same_as(builtinAlloc)) {
					if (auto out = outputOf.find(var); out != outputOf.end())
						values[var] = Value::tensor(out->second); // returned tensor: compute into the output buffer
					else
						values[var] = Value::tensor(allocate(shapeArg(call->args[0]), dtypeArg(call->args[1]), "tensor"));
				} else if (call->op.same_as(allocStorage)) {
					// Storage of `size` bytes; allocated as elements of the storage dtype.
					Array<PrimExpr> size = shapeArg(call->args[0]);
					DataType dtype = dtypeArg(call->args[3]);
					PrimExpr bytes = ::tvm::cast(DataType::Int(64), size[0]);
					PrimExpr elems = ::tvm::floordiv(bytes + (dtype.bytes() - 1), dtype.bytes());
					values[var] = Value::tensor(allocate({elems}, dtype, "storage"));
				} else if (call->op.same_as(allocTensor)) {
					Value storage = evaluate(call->args[0]);
					DataType dtype = dtypeArg(call->args[3]);
					if (storage.kind != Value::Kind::Tensor || storage.buffer->dtype != dtype)
						unsupported("alloc_tensor on storage of another dtype");
					PrimExpr offset = ::tvm::floordiv(
							::tvm::cast(DataType::Int(64), evaluate(call->args[1]).prim), dtype.bytes()
					);
					values[var] = Value::tensor(view(storage.buffer, shapeArg(call->args[2]), dtype, offset));
				} else if (call->op.same_as(reshape)) {
					Value x = evaluate(call->args[0]);
					if (x.kind != Value::Kind::Tensor)
						unsupported("reshape of a non-tensor");
					values[var] = Value::tensor(view(x.buffer, shapeArg(call->args[1]), x.buffer->dtype, x.buffer->elem_offset));
				} else if (call->op.same_as(killStorage) || call->op.same_as(killTensor)) {
					// Lifetimes are handled by the workspace allocations.
				} else {
					unsupported("operation " + std::string(::tvm::Downcast<::tvm::Op>(call->op)->name));
				}
			}
		};
	} // namespace

	namespace {
		/** @brief `size(dims...) -> bytes`: the size of a context buffer for the given dimensions of `...`. */
		tirx::PrimFunc contextSizeFunction(const PrimExpr& bytes, const Array<tirx::Var>& dims, const std::string& symbol) {
			for (const auto& var : tirx::UndefinedVars(bytes))
				if (std::find_if(dims.begin(), dims.end(), [&](const tirx::Var& d) { return d.same_as(var); }) ==
					dims.end())
					unsupported(
							"context buffer layout (its size depends on " + std::string(var->name_hint) +
							", which is not a dimension of `...`)"
					);
			auto attrs = ::tvm::DictAttrs(
					{{::tvm::attr::kGlobalSymbol, ::tvm::ffi::String(symbol)}, {kAOTEntryAttr, ::tvm::Bool(true)}}
			);
			auto body = tirx::Evaluate(tirx::Call(DataType::Int(64), tirx::builtin::ret(), {bytes}));
			return tirx::PrimFunc(dims, body, ::tvm::PrimType(DataType::Int(64)), {}, attrs);
		}
	} // namespace

	IRModule lowerToAOT(
			IRModule mod, const std::string& kernelPrefix, const std::unordered_map<std::string, EntrySignature>& packing,
			bool profile
	) {
		// Give every kernel a unique global symbol.
		std::unordered_map<const ::tvm::GlobalVarNode*, std::string> symbols;
		IRModule result = IRModule(::tvm::ffi::Map<GlobalVar, ::tvm::BaseFunc>(), mod->source_map, mod->attrs, mod->global_infos);
		for (const auto& [gv, func] : mod->functions) {
			if (const auto* prim = func.as<tirx::PrimFuncNode>()) {
				std::string symbol = kernelPrefix + std::string(gv->name_hint);
				symbols[gv.get()] = symbol;
				result->Add(gv, ::tvm::WithAttr(::tvm::ffi::GetRef<tirx::PrimFunc>(prim), ::tvm::attr::kGlobalSymbol,
												::tvm::ffi::String(symbol)));
			}
		}
		for (const auto& [gv, func] : mod->functions) {
			if (const auto* fn = func.as<relax::FunctionNode>()) {
				auto symbol = fn->GetAttr<::tvm::ffi::String>(::tvm::attr::kGlobalSymbol);
				if (!symbol)
					continue; // internal helper function
				EntryBuilder builder{mod, symbols, profile};
				auto packed = packing.find(std::string(symbol.value()));
				const EntrySignature* signature = packed == packing.end() ? nullptr : &packed->second;
				auto entry = builder.build(::tvm::ffi::GetRef<relax::Function>(fn), symbol.value(), signature);
				result->Add(GlobalVar(symbol.value()), entry);
				if (signature != nullptr && !signature->sizeSymbol.empty())
					result->Add(
							GlobalVar(signature->sizeSymbol),
							contextSizeFunction(builder.contextBytes(), signature->dims, signature->sizeSymbol)
					);
			}
		}
		return result;
	}

} // namespace mdl::backend::tvm
