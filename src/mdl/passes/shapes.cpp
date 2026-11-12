#include <mdl/passes/shapes.hpp>

#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/utils/overload.hpp>

#include <algorithm>
#include <format>
#include <functional>
#include <utility>

namespace mdl::shapes {

	Dim Dim::operator*(const Dim& other) const {
		if (!known || !other.known)
			return unknown();
		Dim result{.factor = factor * other.factor, .symbols = symbols, .known = true};
		result.symbols.insert(result.symbols.end(), other.symbols.begin(), other.symbols.end());
		std::sort(result.symbols.begin(), result.symbols.end());
		return result;
	}

	Dim Dim::operator/(const Dim& other) const {
		if (!known || !other.known || other.factor == 0 || factor % other.factor != 0)
			return unknown();
		Dim result{.factor = factor / other.factor, .symbols = symbols, .known = true};
		for (const auto& s : other.symbols) {
			auto it = std::find(result.symbols.begin(), result.symbols.end(), s);
			if (it == result.symbols.end())
				return unknown();
			result.symbols.erase(it);
		}
		return result;
	}

	std::string Dim::str() const {
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

		/** @brief Thrown (internally) when a shape cannot be inferred. */
		struct ShapeError {
			std::string message;
		};

		[[noreturn]] void fail(std::string message) { throw ShapeError{std::move(message)}; }

		Shape product(const Shape& shape, size_t from, size_t to) {
			Dim d = Dim::constant(1);
			for (size_t i = from; i < to; ++i)
				d = d * shape[i];
			return {d};
		}

		Dim numel(const Shape& shape) { return product(shape, 0, shape.size())[0]; }

		/** @brief NumPy broadcasting of two shapes. */
		Shape broadcast(const Shape& a, const Shape& b) {
			const size_t rank = std::max(a.size(), b.size());
			Shape result(rank);
			for (size_t i = 0; i < rank; ++i) {
				const Dim* da = i < rank - a.size() ? nullptr : &a[i - (rank - a.size())];
				const Dim* db = i < rank - b.size() ? nullptr : &b[i - (rank - b.size())];
				if (da == nullptr || *da == Dim::constant(1))
					result[i] = db ? *db : *da;
				else if (db == nullptr || *db == Dim::constant(1) || *da == *db)
					result[i] = *da;
				else if (!da->known || !db->known)
					result[i] = Dim::unknown();
				else
					fail(std::format("cannot broadcast {} and {}", str(a), str(b)));
			}
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

		class Inference {
		public:
			Inference(unsigned ellipsisRank, const ConstantEvaluator& evaluate) : evaluate(evaluate) {
				for (unsigned i = 0; i < ellipsisRank; ++i)
					result.ellipsis.push_back(ellipsisRank == 1 ? std::string{"B"} : "B" + std::to_string(i));
			}

			Shapes result;

			const Shape& shapeOf(const Node& node) {
				if (auto it = result.of.find(&node); it != result.of.end())
					return it->second;
				Shape shape = std::visit([&](const auto& n) { return infer(n, node); }, node);
				return result.of.emplace(&node, std::move(shape)).first->second;
			}

		private:
			const ConstantEvaluator& evaluate;

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

			int64_t integer(const Node& node) {
				auto value = evaluate(node);
				if (!value)
					fail("expected a compile-time integer");
				return *value;
			}

			static const std::vector<int>* vectorConstant(const Node& node) {
				const auto* c = std::get_if<ConstNode>(&node);
				return c ? std::get_if<std::vector<int>>(&c->value) : nullptr;
			}

			Shape infer(const ConstNode& c, const Node&) {
				if (const auto* v = std::get_if<std::vector<int>>(&c.value))
					return {Dim::constant(static_cast<int64_t>(v->size()))};
				return {};
			}
			Shape infer(const ErrorNode&, const Node&) { fail("error node"); }
			Shape infer(const InputNode& in, const Node&) { return fromType(in.type); }
			Shape infer(const ParameterNode& p, const Node&) { return fromType(p.type); }
			Shape infer(const HyperparameterNode&, const Node&) { return {}; }

			Shape infer(const InvocationNode& inv, const Node&) {
				auto arg = [&](size_t i) -> const Shape& {
					if (i >= inv.arguments.size())
						fail("missing argument");
					return shapeOf(inv.arguments[i].get());
				};
				switch (inv.id) {
				case FIDs::TENSOR_ADD:
				case FIDs::TENSOR_SUB:
				case FIDs::TENSOR_PROD:
				case FIDs::TENSOR_DIV:
				case FIDs::EQUAL_MASK:
					return broadcast(arg(0), arg(1));

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
					return arg(0);

				case FIDs::TENSOR_DOT: return matmul(arg(0), arg(1));

				// Reductions over the last axis: max/min/sum keep it (size 1), mean removes it.
				case FIDs::MAX:
				case FIDs::MIN:
				case FIDs::SUM: {
					Shape s = arg(0);
					if (s.empty())
						fail("reduction of a scalar");
					s.back() = Dim::constant(1);
					return s;
				}
				case FIDs::MEAN: {
					Shape s = arg(0);
					if (s.empty())
						fail("reduction of a scalar");
					s.pop_back();
					return s;
				}
				case FIDs::SUM_ALL:
				case FIDs::MEAN_ALL:
					return {};

				case FIDs::RESHAPE: return reshape(inv);
				case FIDs::PERMUTE: {
					const Shape& x = arg(0);
					const auto* perm = vectorConstant(inv.arguments.at(1).get());
					if (perm == nullptr)
						fail("permutation is not a constant");
					auto axes = permuteAxes(static_cast<int64_t>(x.size()), *perm);
					if (axes.empty())
						return x; // invalid permutations are identities (as in the backend)
					Shape s;
					for (int64_t a : axes)
						s.push_back(x[a]);
					return s;
				}
				case FIDs::PERMUTE_INVERSE: {
					const Shape& x = arg(0);
					const auto* perm = vectorConstant(inv.arguments.at(1).get());
					if (perm == nullptr)
						fail("permutation is not a constant");
					auto axes = permuteAxes(static_cast<int64_t>(x.size()), *perm);
					if (axes.empty())
						return x;
					Shape s(x.size());
					for (size_t i = 0; i < axes.size(); ++i)
						s[axes[i]] = x[i];
					return s;
				}
				case FIDs::EXPAND: {
					const Shape& x = arg(0);
					Shape s;
					for (size_t i = 1; i < inv.arguments.size(); ++i) {
						const int64_t d = integer(inv.arguments[i].get());
						s.push_back(d == -1 && i - 1 < x.size() ? x[i - 1] : Dim::constant(d));
					}
					return s;
				}

				case FIDs::COLLAPSE_SUM_LIKE:
				case FIDs::BROADCAST_LIKE:
				case FIDs::RESHAPE_LIKE:
				case FIDs::ONE_HOT_LIKE:
					arg(0);
					return arg(1);
				case FIDs::SCATTER_ADD_ROWS:
					arg(0);
					arg(1);
					return arg(2);

				case FIDs::TAKE: {
					const Shape& table = arg(0);
					if (table.empty())
						fail("take from a scalar");
					Shape s = arg(1);
					s.insert(s.end(), table.begin() + 1, table.end());
					return s;
				}

				default:
					fail(std::format("no shape rule for builtin #{}", std::to_underlying(inv.id)));
				}
			}

			/** @brief NumPy matmul (a 1-D operand is a vector), with broadcasting of the batch dimensions. */
			static Shape matmul(const Shape& a, const Shape& b) {
				if (a.empty() || b.empty())
					fail("matmul of a scalar");
				if (b.size() == 1) {
					Shape s(a.begin(), a.end() - 1);
					return s;
				}
				if (a.size() == 1) {
					Shape s(b.begin(), b.end() - 2);
					s.push_back(b.back());
					return s;
				}
				Shape batch = broadcast(Shape(a.begin(), a.end() - 2), Shape(b.begin(), b.end() - 2));
				batch.push_back(a[a.size() - 2]);
				batch.push_back(b.back());
				return batch;
			}

			/**
			 * @brief RESHAPE: one entry per output dimension: a size, -1 (the next leading input dimension), -2 (inferred),
			 * or an ellipsis group {before, after} (the input dimensions [before, rank - after)).
			 */
			Shape reshape(const InvocationNode& inv) {
				const Shape x = shapeOf(inv.arguments.at(0).get());
				Shape s;
				std::optional<size_t> inferred;
				size_t passthrough = 0;
				for (size_t i = 1; i < inv.arguments.size(); ++i) {
					if (const auto* group = vectorConstant(inv.arguments[i].get()); group && group->size() == 2) {
						for (int64_t d = (*group)[0]; d < static_cast<int64_t>(x.size()) - (*group)[1]; ++d)
							s.push_back(x[d]);
						continue;
					}
					const int64_t d = integer(inv.arguments[i].get());
					if (d == -1) {
						s.push_back(passthrough < x.size() ? x[passthrough] : Dim::unknown());
						++passthrough;
					} else if (d == -2) {
						inferred = s.size();
						s.push_back(Dim::unknown());
					} else {
						s.push_back(Dim::constant(d));
					}
				}
				if (inferred) {
					Dim others = Dim::constant(1);
					for (size_t i = 0; i < s.size(); ++i)
						if (i != *inferred)
							others = others * s[i];
					s[*inferred] = numel(x) / others;
				}
				return s;
			}
		};
	} // namespace

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
