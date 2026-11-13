#include <mdl/passes/shapes.hpp>

#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/utils/overload.hpp>

#include <algorithm>
#include <format>
#include <functional>
#include <utility>

namespace mdl::shapes {

	Dim Dim::operator*(const Dim& other) const {
		if (other.isConstant() && other.factor == 1)
			return *this;
		if (isConstant() && factor == 1)
			return other;
		if (!known || !other.known || group || other.group || node != nullptr || other.node != nullptr)
			return unknown();
		Dim result{.factor = factor * other.factor, .symbols = symbols};
		result.symbols.insert(result.symbols.end(), other.symbols.begin(), other.symbols.end());
		std::sort(result.symbols.begin(), result.symbols.end());
		return result;
	}

	Dim Dim::operator/(const Dim& other) const {
		if (other.isConstant() && other.factor == 1)
			return *this;
		if (!known || !other.known || group || other.group || node != nullptr || other.node != nullptr ||
			other.factor == 0 || factor % other.factor != 0)
			return unknown();
		Dim result{.factor = factor / other.factor, .symbols = symbols};
		for (const auto& s : other.symbols) {
			auto it = std::find(result.symbols.begin(), result.symbols.end(), s);
			if (it == result.symbols.end())
				return unknown();
			result.symbols.erase(it);
		}
		return result;
	}

	std::string Dim::str() const {
		if (group)
			return "...";
		if (node != nullptr)
			return "N";
		if (!known)
			return "?";
		std::string s = (factor != 1 || symbols.empty()) ? std::to_string(factor) : "";
		for (const auto& sym : symbols)
			s += (s.empty() ? "" : "*") + sym;
		return s;
	}

	std::string str(const Shape& shape) {
		std::string s = "[";
		for (size_t i = 0; i < shape.size(); ++i)
			s += (i > 0 ? ", " : "") + shape[i].str();
		return s + "]";
	}

	namespace {
		using FIDs = builtins::FunctionIDs;
		using NodeRef = std::reference_wrapper<const Node>;

		/** @brief Thrown (internally) when a shape cannot be inferred. */
		struct ShapeError {
			std::string message;
		};

		/** @brief A definite mismatch of the arguments (e.g. of the inner dimensions of a matrix product). */
		struct Mismatch : ShapeError {};

		[[noreturn]] void fail(std::string message) { throw ShapeError{std::move(message)}; }
		[[noreturn]] void mismatched(std::string message) { throw Mismatch{{std::move(message)}}; }

		Dim numel(const Shape& shape) {
			Dim d = Dim::constant(1);
			for (const auto& dim : shape)
				d = d * dim;
			return d;
		}

		bool hasGroup(const Shape& s) {
			return std::any_of(s.begin(), s.end(), [](const Dim& d) { return d.group; });
		}

		/**
		 * @brief NumPy broadcasting of two shapes. With groups (rank-polymorphic types), the dimensions are aligned from
		 * the end up to the first group, which takes the remaining dimensions of the other shape.
		 */
		Shape broadcast(const Shape& a, const Shape& b, bool symbolic) {
			Shape tail;
			size_t k = 1;
			for (; k <= std::min(a.size(), b.size()); ++k) {
				const Dim& da = a[a.size() - k];
				const Dim& db = b[b.size() - k];
				if (da.group || db.group) {
					const Shape& rest = da.group ? a : b;
					Shape result(rest.begin(), rest.end() - static_cast<std::ptrdiff_t>(k - 1));
					result.insert(result.end(), tail.rbegin(), tail.rend());
					return result;
				}
				if (da == Dim::constant(1))
					tail.push_back(db);
				else if (db == Dim::constant(1) || da == db)
					tail.push_back(da);
				else if (!da.known || !db.known || (symbolic && !(da.isConstant() && db.isConstant())))
					tail.push_back(Dim::unknown());
				else
					mismatched(std::format("cannot broadcast {} and {}", str(a), str(b)));
			}
			const Shape& longer = a.size() >= b.size() ? a : b;
			Shape result(longer.begin(), longer.end() - static_cast<std::ptrdiff_t>(k - 1));
			result.insert(result.end(), tail.rbegin(), tail.rend());
			return result;
		}

		/** @brief The axes of a PERMUTE permutation (see builtins::FunctionIDs::PERMUTE); empty if invalid. */
		std::vector<int64_t> permuteAxes(int64_t rank, const std::vector<int>& perm) {
			const int64_t size = static_cast<int64_t>(perm.size());
			std::vector<int64_t> axes;
			if (std::find(perm.begin(), perm.end(), builtins::kEllipsisPermMarker) != perm.end()) {
				const int64_t start = std::count_if(perm.begin(), perm.end(), [](int p) { return p >= 0; });
				const int64_t groupRank = rank - (size - 1);
				for (int p : perm) {
					if (p == builtins::kEllipsisPermMarker)
						for (int64_t d = 0; d < groupRank; ++d)
							axes.push_back(start + d);
					else
						axes.push_back(p < 0 ? p + rank : p);
				}
			} else {
				for (int64_t i = 0; i < rank - size; ++i)
					axes.push_back(i);
				for (int p : perm)
					axes.push_back(p < 0 && rank > 0 ? p + rank : p);
			}
			std::vector<bool> seen(std::max<int64_t>(rank, 0), false);
			if (static_cast<int64_t>(axes.size()) != rank)
				return {};
			for (int64_t a : axes) {
				if (a < 0 || a >= rank || seen[a])
					return {};
				seen[a] = true;
			}
			return axes;
		}

		const std::vector<int>* vectorConstant(const Node& node) {
			const auto* c = std::get_if<ConstNode>(&node);
			return c ? std::get_if<std::vector<int>>(&c->value) : nullptr;
		}

		/**
		 * @brief The shape rules of the builtins, shared by the exact inference for a fixed rank of `...` (infer) and
		 * the rank-polymorphic types of the computegraph (resultType, `symbolic`).
		 * @details The rules match the semantics of the builtins as the backends implement them (e.g. `max` and `sum`
		 * keep the reduced axis with size 1, `mean` removes it). In symbolic mode, shapes may contain groups (`...`,
		 * any number of dimensions) and non-constant compile-time dimensions; what a rule cannot express exactly
		 * becomes a group or an unknown dimension instead of an error.
		 */
		struct Rules {
			const ConstantEvaluator& evaluate;
			bool symbolic;

			int64_t integer(const Node& node) const {
				auto value = evaluate(node);
				if (!value)
					fail("expected a compile-time integer");
				return *value;
			}

			/** @brief A size given by a compile-time integer (symbolic: also a not yet constant one). */
			Dim size(const Node& node) const {
				if (auto value = evaluate(node))
					return Dim::constant(*value);
				if (symbolic)
					return Dim::of(node);
				fail("expected a compile-time integer");
			}

			Shape apply(FIDs id, const std::vector<NodeRef>& args, const std::function<const Shape&(size_t)>& shapeOf) const {
				auto arg = [&](size_t i) -> const Shape& {
					if (i >= args.size())
						fail("missing argument");
					return shapeOf(i);
				};
				switch (id) {
				case FIDs::TENSOR_ADD:
				case FIDs::TENSOR_SUB:
				case FIDs::TENSOR_PROD:
				case FIDs::TENSOR_DIV:
				case FIDs::EQUAL_MASK:
					return broadcast(arg(0), arg(1), symbolic);
				case FIDs::WHERE:
					return broadcast(arg(0), broadcast(arg(1), arg(2), symbolic), symbolic);

				case FIDs::COS:
				case FIDs::SIN:
				case FIDs::TAN:
				case FIDs::EXP:
				case FIDs::ERF:
				case FIDs::SQRT:
				case FIDs::RSQRT:
				case FIDs::LOG:
				case FIDs::TANH:
				case FIDs::STOP_GRADIENT:
				case FIDs::MASK_OF:
				case FIDs::UNMASK:
					return arg(0);

				case FIDs::TENSOR_DOT: return matmul(arg(0), arg(1));

				// Reductions over the last axis: max/min/sum keep it (size 1), mean removes it. (If the last axis is
				// part of a group, the group keeps the others.)
				case FIDs::MAX:
				case FIDs::MIN:
				case FIDs::SUM: {
					Shape s = arg(0);
					if (s.empty())
						fail("reduction of a scalar");
					if (s.back().group)
						s.push_back(Dim::constant(1));
					else
						s.back() = Dim::constant(1);
					return s;
				}
				case FIDs::MEAN: {
					Shape s = arg(0);
					if (s.empty())
						fail("reduction of a scalar");
					if (!s.back().group)
						s.pop_back();
					return s;
				}
				case FIDs::SUM_ALL:
				case FIDs::MEAN_ALL:
				case FIDs::VAR:
					arg(0); // (the shapes of the whole function are inferred)
					return {};

				case FIDs::RESHAPE: return reshape(args, arg(0));
				case FIDs::PERMUTE:
				case FIDs::PERMUTE_INVERSE: {
					const Shape& x = arg(0);
					const auto* perm = vectorConstant(args.at(1).get());
					if (perm == nullptr)
						fail("permutation is not a constant");
					auto axes = permuteAxes(static_cast<int64_t>(x.size()), *perm);
					if (axes.empty())
						return x; // invalid permutations are identities (as in the backend)
					Shape s(x.size());
					for (size_t i = 0; i < axes.size(); ++i) {
						if (id == FIDs::PERMUTE)
							s[i] = x[axes[i]];
						else
							s[axes[i]] = x[i];
					}
					return s;
				}
				case FIDs::EXPAND: {
					const Shape& x = arg(0);
					Shape s;
					for (size_t i = 1; i < args.size(); ++i) {
						Dim d = size(args[i].get());
						s.push_back(d == Dim::constant(-1) && i - 1 < x.size() ? x[i - 1] : d);
					}
					return s;
				}

				case FIDs::COLLAPSE_SUM_LIKE:
				case FIDs::BROADCAST_LIKE:
				case FIDs::RESHAPE_LIKE:
					arg(0);
					return arg(1);
				case FIDs::ONE_HOT_LIKE: {
					// The labels have the shape of `like` without its last dimension (the classes).
					const Shape& labels = arg(0);
					const Shape& like = arg(1);
					if (!symbolic && (like.empty() || labels != Shape(like.begin(), like.end() - 1)))
						fail(std::format(
								"the labels {} do not match the classes {} (one label per row; is the rank of the "
								"labels' `...` right? see --ellipsis-rank NAME=K)",
								str(labels), str(like)
						));
					return like;
				}
				case FIDs::SCATTER_ADD_ROWS:
					arg(0);
					arg(1);
					return arg(2);

				case FIDs::SELECT: {
					Shape s = arg(0);
					const int64_t rank = static_cast<int64_t>(s.size());
					const int64_t axis = integer(args.at(1).get());
					const int64_t a = axis < 0 ? axis + rank : axis;
					if (a < 0 || a >= rank)
						fail("select: axis out of range");
					// (The axis is counted from the front or the end; a group in between takes the removal.)
					for (int64_t k = axis < 0 ? a : 0; k < (axis < 0 ? rank : a + 1); ++k)
						if (s[k].group)
							return s;
					s.erase(s.begin() + a);
					return s;
				}
				case FIDs::UNSELECT_LIKE:
					arg(0);
					return arg(3);

				case FIDs::TAKE: {
					const Shape& table = arg(0);
					if (table.empty())
						fail("take from a scalar");
					Shape s = arg(1);
					s.insert(s.end(), table.begin() + 1, table.end());
					return s;
				}

				default:
					fail(std::format("no shape rule for builtin #{}", std::to_underlying(id)));
				}
			}

			/** @brief NumPy matmul (a 1-D operand is a vector), with broadcasting of the batch dimensions. */
			Shape matmul(const Shape& a, const Shape& b) const {
				if (a.empty() || b.empty())
					fail("matmul of a scalar");
				const Dim& ka = a.back();
				const Dim& kb = b.size() == 1 ? b.back() : b[b.size() - 2];
				if (ka.known && kb.known && !ka.group && !kb.group && ka.node == nullptr && kb.node == nullptr && ka != kb)
					mismatched(std::format("the inner dimensions of the matrix product differ ({} @ {})", str(a), str(b)));
				if (b.size() == 1)
					return Shape(a.begin(), a.end() - 1);
				if (a.size() == 1) {
					Shape s(b.begin(), b.end() - 2);
					s.push_back(b.back());
					return s;
				}
				// A group among the matrix dimensions: a[..., m, k] @ b[k, n] with m (and batch dims) in the group.
				if (a[a.size() - 2].group || b[b.size() - 2].group || a.back().group || b.back().group) {
					Shape s(a.begin(), a.end() - 1);
					s.push_back(b.back());
					return s;
				}
				Shape batch = broadcast(Shape(a.begin(), a.end() - 2), Shape(b.begin(), b.end() - 2), symbolic);
				batch.push_back(a[a.size() - 2]);
				batch.push_back(b.back());
				return batch;
			}

			/**
			 * @brief RESHAPE: one entry per output dimension: a size, -1 (the next leading input dimension), -2 (inferred),
			 * or an ellipsis group {before, after} (the input dimensions [before, rank - after)).
			 */
			Shape reshape(const std::vector<NodeRef>& args, const Shape& x) const {
				Shape s;
				std::optional<size_t> inferred;
				size_t passthrough = 0;
				bool afterGroup = false; // passthrough dimensions behind a group are not known
				for (size_t i = 1; i < args.size(); ++i) {
					if (const auto* group = vectorConstant(args[i].get()); group && group->size() == 2) {
						for (int64_t d = (*group)[0]; d < static_cast<int64_t>(x.size()) - (*group)[1]; ++d)
							s.push_back(x[d]);
						continue;
					}
					const Dim d = size(args[i].get());
					if (d == Dim::constant(-1)) {
						afterGroup |= passthrough < x.size() && x[passthrough].group;
						s.push_back(passthrough < x.size() && !afterGroup ? x[passthrough] : Dim::unknown());
						++passthrough;
					} else if (d == Dim::constant(-2)) {
						inferred = s.size();
						s.push_back(Dim::unknown());
					} else {
						s.push_back(d);
					}
				}
				if (inferred) {
					Dim others = Dim::constant(1);
					for (size_t i = 0; i < s.size(); ++i)
						if (i != *inferred)
							others = others * s[i];
					s[*inferred] = numel(x) / others;
					if (s[*inferred].group)
						s[*inferred] = Dim::unknown();
				}
				return s;
			}
		};

		class Inference {
		public:
			Inference(unsigned ellipsisRank, const ConstantEvaluator& evaluate)
					: evaluate(evaluate), rules{evaluate, false} {
				for (unsigned i = 0; i < ellipsisRank; ++i)
					result.ellipsis.push_back(ellipsisRank == 1 ? std::string{"B"} : "B" + std::to_string(i));
			}

			Shapes result;

			const Shape& shapeOf(const Node& node) {
				if (auto it = result.of.find(&node); it != result.of.end())
					return it->second;
				Shape shape = std::visit([&](const auto& n) { return infer(n); }, node);
				return result.of.emplace(&node, std::move(shape)).first->second;
			}

		private:
			const ConstantEvaluator& evaluate;
			Rules rules;

			Shape fromType(const DataType& type) {
				if (std::holds_alternative<PrimitiveType>(type))
					return {};
				const auto* tensor = std::get_if<TensorType>(&type);
				if (tensor == nullptr)
					fail("not a tensor or scalar");
				Shape shape;
				for (const auto& dim : tensor->shape) {
					std::visit(
							overloaded{
									[&](Elipsis) {
										for (const auto& s : result.ellipsis)
											shape.push_back(Dim::symbol(s));
									},
									[&](unsigned u) { shape.push_back(Dim::constant(u)); },
									[&](const Node* n) {
										auto value = evaluate(*n);
										shape.push_back(value ? Dim::constant(*value) : Dim::unknown());
									},
							},
							dim
					);
				}
				return shape;
			}

			Shape infer(const ConstNode& c) {
				if (const auto* v = std::get_if<std::vector<int>>(&c.value))
					return {Dim::constant(static_cast<int64_t>(v->size()))};
				return {};
			}
			Shape infer(const ErrorNode&) { fail("error node"); }
			Shape infer(const InputNode& in) { return fromType(in.type); }
			Shape infer(const ParameterNode& p) { return fromType(p.type); }
			Shape infer(const HyperparameterNode&) { return {}; }
			Shape infer(const InvocationNode& inv) {
				return rules.apply(inv.id, inv.arguments, [&](size_t i) -> const Shape& {
					return shapeOf(inv.arguments[i].get());
				});
			}
		};

		/** @brief The element type of the result of @p id (see resultType). */
		PrimitiveType elementType(FIDs id, const std::vector<NodeRef>& args) {
			auto of = [&](size_t i) -> std::optional<PrimitiveType> {
				if (i >= args.size())
					return std::nullopt;
				const DataType& t = getDataType(args[i].get());
				if (const auto* p = std::get_if<PrimitiveType>(&t))
					return *p;
				if (const auto* tensor = std::get_if<TensorType>(&t))
					return tensor->type;
				return std::nullopt;
			};
			auto isTensor = [&](size_t i) { return i < args.size() && std::holds_alternative<TensorType>(getDataType(args[i].get())); };
			switch (id) {
			case FIDs::MASK_OF:
			case FIDs::SUM_ALL:
			case FIDs::MEAN_ALL:
			case FIDs::EQUAL_MASK:
			case FIDs::ONE_HOT_LIKE:
				return PrimitiveType::floating;
			case FIDs::WHERE:
				return of(1).value_or(PrimitiveType::floating);
			case FIDs::COLLAPSE_SUM_LIKE:
			case FIDs::BROADCAST_LIKE:
			case FIDs::RESHAPE_LIKE:
				return of(1).value_or(PrimitiveType::floating);
			case FIDs::SCATTER_ADD_ROWS:
				return of(2).value_or(PrimitiveType::floating);
			case FIDs::UNSELECT_LIKE:
				return of(3).value_or(PrimitiveType::floating);
			case FIDs::TENSOR_ADD:
			case FIDs::TENSOR_SUB:
			case FIDs::TENSOR_PROD:
			case FIDs::TENSOR_DIV:
				// The tensor operand's type; scalar arithmetic is floating if an operand is.
				for (size_t i = 0; i < args.size(); ++i)
					if (isTensor(i))
						return *of(i);
				for (size_t i = 0; i < args.size(); ++i)
					if (of(i) == PrimitiveType::floating)
						return PrimitiveType::floating;
				return of(0).value_or(PrimitiveType::floating);
			case FIDs::COS:
			case FIDs::SIN:
			case FIDs::TAN:
			case FIDs::EXP:
			case FIDs::ERF:
			case FIDs::SQRT:
			case FIDs::RSQRT:
			case FIDs::LOG:
			case FIDs::TANH:
				return isTensor(0) ? *of(0) : PrimitiveType::floating;
			default:
				return of(0).value_or(PrimitiveType::floating);
			}
		}

		/** @brief Constant integers (the sizes and axes of the builtins in the computegraph's types). */
		std::optional<int64_t> constantInteger(const Node& node) {
			if (const auto* c = std::get_if<ConstNode>(&node))
				if (const auto* i = std::get_if<int>(&c->value))
					return *i;
			return std::nullopt;
		}

	} // namespace

	std::optional<Shape> shapeOfType(const DataType& type) {
			if (std::holds_alternative<PrimitiveType>(type))
				return Shape{};
			const auto* tensor = std::get_if<TensorType>(&type);
			if (tensor == nullptr)
				return std::nullopt;
			Shape shape;
			for (const auto& dim : tensor->shape)
				std::visit(
						overloaded{
								[&](Elipsis) { shape.push_back(Dim::ellipsis()); },
								[&](unsigned u) { shape.push_back(Dim::constant(u)); },
								[&](const Node* n) {
									auto value = constantInteger(*n);
									shape.push_back(value ? Dim::constant(*value) : Dim::of(*n));
								},
						},
						dim
				);
			return shape;
	}

	namespace {
		TensorType::Shape typeShape(const Shape& shape) {
			TensorType::Shape result;
			for (const auto& d : shape) {
				if (d.isConstant() && d.factor >= 0)
					result.emplace_back(static_cast<unsigned>(d.factor));
				else if (d.node != nullptr)
					result.emplace_back(d.node);
				else
					result.emplace_back(Elipsis{}); // a group or an unknown size
			}
			return result;
		}
	} // namespace

	std::string str(const DataType& type) {
		if (const auto* p = std::get_if<PrimitiveType>(&type))
			return *p == PrimitiveType::floating ? "float" : *p == PrimitiveType::boolean ? "bool" : "int";
		const auto* t = std::get_if<TensorType>(&type);
		if (t == nullptr)
			return "?";
		std::string s = (t->masked ? "masked " : "") + str(DataType{t->type});
		auto shape = shapeOfType(type);
		return s + (shape ? str(*shape) : "[?]");
	}

	bool matches(const DataType& declared, const DataType& computed, bool elements) {
		auto element = [](const DataType& t) -> std::optional<PrimitiveType> {
			if (const auto* p = std::get_if<PrimitiveType>(&t))
				return *p;
			if (const auto* tensor = std::get_if<TensorType>(&t))
				return tensor->type;
			return std::nullopt;
		};
		const auto de = element(declared), ce = element(computed);
		if (!de || !ce || *de == PrimitiveType::error || *ce == PrimitiveType::error)
			return true;
		if (elements && (*de == PrimitiveType::floating) != (*ce == PrimitiveType::floating))
			return false;
		const auto ds = shapeOfType(declared), cs = shapeOfType(computed);
		if (!ds || !cs || !std::holds_alternative<TensorType>(computed))
			return true;
		if (!std::holds_alternative<TensorType>(declared)) // a declared scalar
			return cs->empty() || hasGroup(*cs);
		// From the end up to the first group of either shape.
		for (size_t k = 1; k <= std::max(ds->size(), cs->size()); ++k) {
			if (k > ds->size() || k > cs->size()) {
				const Shape& longer = ds->size() > cs->size() ? *ds : *cs;
				return longer[longer.size() - k].group;
			}
			const Dim& d = (*ds)[ds->size() - k];
			const Dim& c = (*cs)[cs->size() - k];
			if (d.group || c.group)
				return true;
			if (d.isConstant() && c.isConstant() && d != c)
				return false;
		}
		return true;
	}


	std::optional<std::string> check(builtins::FunctionIDs id, const std::vector<std::reference_wrapper<const Node>>& args) {
		std::vector<Shape> shapes;
		for (const auto& arg : args) {
			auto shape = shapeOfType(getDataType(arg.get()));
			if (!shape)
				return std::nullopt;
			shapes.push_back(std::move(*shape));
		}
		static const ConstantEvaluator evaluate = constantInteger;
		try {
			Rules{evaluate, true}.apply(id, args, [&](size_t i) -> const Shape& { return shapes[i]; });
		} catch (const Mismatch& e) {
			return e.message;
		} catch (const ShapeError&) {
		}
		return std::nullopt;
	}

	DataType resultType(builtins::FunctionIDs id, const std::vector<std::reference_wrapper<const Node>>& args) {
		const PrimitiveType element = elementType(id, args);
		const DataType unknown = TensorType{element, {Elipsis{}}};
		std::vector<Shape> shapes;
		bool anyTensor = false;
		for (const auto& arg : args) {
			const DataType& type = getDataType(arg.get());
			anyTensor |= std::holds_alternative<TensorType>(type);
			auto shape = shapeOfType(type);
			if (!shape)
				return unknown;
			shapes.push_back(std::move(*shape));
		}
		static const ConstantEvaluator evaluate = constantInteger;
		try {
			Shape shape = Rules{evaluate, true}.apply(id, args, [&](size_t i) -> const Shape& { return shapes[i]; });
			if (id == FIDs::SUM_ALL || id == FIDs::MEAN_ALL || (!anyTensor && shape.empty()))
				return element;
			return TensorType{element, typeShape(shape)};
		} catch (const ShapeError&) {
			return unknown;
		}
	}

	struct Oracle::Impl {
		ConstantEvaluator evaluate;
		Inference inference;
		Impl(unsigned rank, ConstantEvaluator e) : evaluate(std::move(e)), inference(rank, evaluate) {}
	};

	Oracle::Oracle(unsigned ellipsisRank, ConstantEvaluator evaluate, std::unordered_map<const Node*, Shape> known)
			: impl(std::make_unique<Impl>(ellipsisRank, std::move(evaluate))) {
		impl->inference.result.of = std::move(known);
	}

	Oracle::~Oracle() = default;

	const Shape* Oracle::shapeOf(const Node& node) {
		try {
			return &impl->inference.shapeOf(node);
		} catch (const ShapeError&) {
			return nullptr;
		}
	}

	void Oracle::assume(const Node& node, Shape shape) { impl->inference.result.of[&node] = std::move(shape); }

	std::expected<Shapes, std::string> infer(
			const FunctionDecl& func, unsigned ellipsisRank, const ConstantEvaluator& evaluate,
			const std::unordered_map<const Node*, Shape>& known
	) {
		Inference inference{ellipsisRank, evaluate};
		inference.result.of = known;
		try {
			for (const auto& output : func.outputs)
				inference.shapeOf(output.get());
		} catch (const ShapeError& e) {
			return std::unexpected(func.name + ": " + e.message);
		}
		return std::move(inference.result);
	}

} // namespace mdl::shapes
