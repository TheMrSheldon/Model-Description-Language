#include "./iree_backend.hpp"

#include "../../argvals.hpp"
#include "../../logging.hpp"

#include <mdl/compiler.hpp>
#include <mdl/computegraphv2/builtins.hpp>
#include <mdl/computegraphv2/datatype.hpp>
#include <mdl/computegraphv2/node.hpp>
#include <mdl/parserv2/compilation_unit.hpp>
#include <mdl/utils/utils.hpp>

#include <cassert>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── type helpers ──────────────────────────────────────────────────────────────

static std::string elemTypeStr(mdl::PrimitiveType pt) {
    switch (pt) {
    case mdl::PrimitiveType::floating: return "f32";
    case mdl::PrimitiveType::integer:  return "i32";
    case mdl::PrimitiveType::uint:     return "ui32";
    case mdl::PrimitiveType::boolean:  return "i1";
    default: return "f32";
    }
}

static std::optional<int64_t> tryEvalConst(const mdl::Node& node, const mdl::ArgVals& argVals) {
    using FIDs = mdl::builtins::FunctionIDs;
    if (const auto* cn = std::get_if<mdl::ConstNode>(&node)) {
        if (const int* v = std::get_if<int>(&cn->value))
            return static_cast<int64_t>(*v);
    } else if (const auto* hn = std::get_if<mdl::HyperparameterNode>(&node)) {
        if (auto v = argVals.lookup(hn->name))
            return static_cast<int64_t>(*v);
    } else if (const auto* inv = std::get_if<mdl::InvocationNode>(&node)) {
        if (inv->arguments.size() == 2) {
            auto lhs = tryEvalConst(inv->arguments[0].get(), argVals);
            auto rhs = tryEvalConst(inv->arguments[1].get(), argVals);
            if (lhs && rhs) {
                switch (inv->id) {
                case FIDs::TENSOR_ADD:  return *lhs + *rhs;
                case FIDs::TENSOR_SUB:  return *lhs - *rhs;
                case FIDs::TENSOR_PROD: return *lhs * *rhs;
                case FIDs::TENSOR_DIV:  return *rhs != 0 ? std::optional<int64_t>(*lhs / *rhs) : std::nullopt;
                default: break;
                }
            }
        }
    }
    return std::nullopt;
}

static std::string dimStr(
    const std::variant<unsigned, mdl::Elipsis, const mdl::Node*>& dim,
    const mdl::ArgVals& argVals)
{
    return std::visit(mdl::overloaded{
        [](mdl::Elipsis)  -> std::string { return "?"; },
        [](unsigned u)    -> std::string { return std::to_string(u); },
        [&](const mdl::Node* n) -> std::string {
            if (auto v = tryEvalConst(*n, argVals))
                return std::to_string(*v);
            auto log = mdl::log::getLogger("iree-backend");
            if (const auto* hn = std::get_if<mdl::HyperparameterNode>(n))
                log->warn("dimStr: unresolved hp '{}'", hn->name);
            else if (std::holds_alternative<mdl::InvocationNode>(*n))
                log->warn("dimStr: unresolved InvocationNode");
            else if (std::holds_alternative<mdl::ConstNode>(*n))
                log->warn("dimStr: unresolved ConstNode");
            else
                log->warn("dimStr: unresolved unknown node type");
            return "?";
        },
    }, dim);
}

static int rankOf(const mdl::DataType& dt) {
    if (const auto* tt = std::get_if<mdl::TensorType>(&dt))
        return static_cast<int>(tt->shape.size());
    return 0;
}

static std::string tensorTypeStr(const mdl::TensorType& tt, const mdl::ArgVals& argVals) {
    std::string s = "tensor<";
    for (size_t i = 0; i < tt.shape.size(); ++i) {
        if (i > 0) s += "x";
        s += dimStr(tt.shape[i], argVals);
    }
    s += "x" + elemTypeStr(tt.type) + ">";
    return s;
}

static std::string mlirTypeStr(const mdl::DataType& dt, const mdl::ArgVals& argVals) {
    return std::visit(mdl::overloaded{
        // StableHLO only operates on tensor types; wrap scalars in 0-rank tensor.
        [](mdl::PrimitiveType pt)       -> std::string { return std::string("tensor<") + elemTypeStr(pt) + ">"; },
        [&](const mdl::TensorType& tt)  -> std::string { return tensorTypeStr(tt, argVals); },
        [](const mdl::AggregateType&)   -> std::string { assert(false); return ""; },
        [](const mdl::ModuleType&)      -> std::string { assert(false); return ""; },
        [](const mdl::BoundModuleType&) -> std::string { assert(false); return ""; },
    }, dt);
}

// ── per-function MLIR translation ─────────────────────────────────────────────

static std::string translateFuncToMlir(
    const mdl::FunctionDecl& func,
    const mdl::ArgVals& argVals,
    std::string_view moduleName)
{
    auto logger = mdl::log::getLogger("iree-backend");
    logger->info("Translating {} (StableHLO)", func.name);

    // Pre-DFS: collect ParameterNodes in visitation order (mirrors tvm.cpp:318–341)
    std::vector<const mdl::Node*> paramNodes;
    std::unordered_set<const mdl::Node*> visited;
    std::function<void(const mdl::Node&)> preTraverse = [&](const mdl::Node& node) {
        if (!visited.insert(&node).second) return;
        if (const auto* inv = std::get_if<mdl::InvocationNode>(&node)) {
            for (const auto& argRef : inv->arguments)
                preTraverse(argRef.get());
        } else if (std::holds_alternative<mdl::ParameterNode>(node)) {
            paramNodes.push_back(&node);
        }
    };
    preTraverse(func.output);

    // Cache: Node* → SSA name; pre-seeded with parameter arg names
    std::unordered_map<const mdl::Node*, std::string> cache;
    for (size_t i = 0; i < paramNodes.size(); ++i)
        cache[paramNodes[i]] = "%arg" + std::to_string(i);

    std::ostringstream body;
    int counter = 0;

    auto freshName = [&](std::string_view hint = "v") {
        return "%" + std::string(hint) + std::to_string(counter++);
    };

    auto bindOp = [&](std::string_view rhs, std::string_view hint = "v") -> std::string {
        auto n = freshName(hint);
        body << "    " << n << " = " << rhs << "\n";
        return n;
    };

    // ── reduce helper ─────────────────────────────────────────────────────────
    auto emitReduce = [&](
        const std::string& input,
        const std::string& inType,
        const std::string& outType,
        const std::string& elemType,
        const std::string& initVal,
        const std::string& reducerOp,
        int axis) -> std::string
    {
        std::string init = bindOp(
            std::format("stablehlo.constant dense<{}> : tensor<{}>", initVal, elemType), "init");
        auto res = freshName("red");
        body << "    " << res
             << " = stablehlo.reduce(" << input << " init: " << init << ")"
             << " across dimensions = [" << axis << "]"
             << " : (" << inType << ", tensor<" << elemType << ">) -> " << outType << "\n"
             << "     reducer(%ra: tensor<" << elemType << ">, %rb: tensor<" << elemType << ">) {\n"
             << "       %rr = " << reducerOp << " %ra, %rb : tensor<" << elemType << ">\n"
             << "       stablehlo.return %rr : tensor<" << elemType << ">\n"
             << "     }\n";
        return res;
    };

    // ── broadcast srcSSA (MDL type srcDT) to match the runtime shape of refSSA ─
    // Uses stablehlo.broadcast_in_dim for fully-static outputs and
    // "stablehlo.dynamic_broadcast_in_dim" (generic form with required IREE
    // attributes) for outputs with Elipsis dimensions.
    // Returns srcSSA unchanged when no broadcast is needed.
    auto broadcastTo = [&](
        const std::string& srcSSA,
        const mdl::DataType& srcDT,
        const std::string& refSSA,   // tensor to read runtime shape from
        const mdl::DataType& refDT   // MDL type of refSSA = desired output type
    ) -> std::string {
        int srcRank = rankOf(srcDT);
        const auto* refTT = std::get_if<mdl::TensorType>(&refDT);
        if (!refTT) return srcSSA;
        int refRank = static_cast<int>(refTT->shape.size());

        bool bc = (srcRank < refRank);
        if (!bc) {
            const auto* srcTT = std::get_if<mdl::TensorType>(&srcDT);
            if (srcTT && srcRank == refRank) {
                for (size_t k = 0; k < srcTT->shape.size() && k < refTT->shape.size(); ++k) {
                    const auto* su = std::get_if<unsigned>(&srcTT->shape[k]);
                    if (!su || *su != 1) continue;
                    const auto* du = std::get_if<unsigned>(&refTT->shape[k]);
                    if (!du || *du != 1) { bc = true; break; }
                }
            }
        }
        if (!bc) return srcSSA;

        // PrimitiveType::integer is emitted as tensor<f32> (see ConstNode handler)
        std::string srcMlir = std::holds_alternative<mdl::PrimitiveType>(srcDT)
            ? "tensor<f32>" : mlirTypeStr(srcDT, argVals);
        std::string refMlir = mlirTypeStr(refDT, argVals);

        // Determine broadcast_dimensions by matching src dims to ref dims.
        // Try leading alignment (src[i] → ref[i]) first; if static dims conflict,
        // fall back to trailing alignment (src[i] → ref[refRank-srcRank+i]).
        // Node* dims (hyperparameters like numHeads) are evaluated via tryEvalConst.
        std::vector<int> bcDimsVec;
        bool leadingAlignment = false;
        if (srcRank == 0) {
            // scalar: no dims to match
        } else if (srcRank == refRank) {
            for (int i = 0; i < srcRank; ++i) bcDimsVec.push_back(i);
            leadingAlignment = true;
        } else {
            const auto* srcTTm = std::get_if<mdl::TensorType>(&srcDT);
            // Returns the evaluated static value of a shape dim, or -1 for Elipsis/unknown.
            auto evalDim = [&](const mdl::TensorType::Shape::value_type& d) -> int64_t {
                if (std::holds_alternative<mdl::Elipsis>(d)) return -1;
                if (const auto* u = std::get_if<unsigned>(&d)) return *u;
                if (const auto* n = std::get_if<const mdl::Node*>(&d))
                    return tryEvalConst(**n, argVals).value_or(-2);
                return -2;
            };
            auto dimMatch = [&](int si, int ri) -> bool {
                const auto& sd = srcTTm->shape[si];
                const auto& rd = refTT->shape[ri];
                bool sdElip = std::holds_alternative<mdl::Elipsis>(sd);
                bool rdElip = std::holds_alternative<mdl::Elipsis>(rd);
                if (sdElip != rdElip) return false;
                if (sdElip) return true;  // both dynamic
                int64_t sv = evalDim(sd), rv = evalDim(rd);
                return sv >= 0 && rv >= 0 && sv == rv;
            };
            bool leadingOk = true;
            for (int i = 0; i < srcRank && leadingOk; ++i)
                if (!dimMatch(i, i)) leadingOk = false;
            if (leadingOk) {
                for (int i = 0; i < srcRank; ++i) bcDimsVec.push_back(i);
                leadingAlignment = true;
            } else {
                // trailing alignment
                for (int i = 0; i < srcRank; ++i)
                    bcDimsVec.push_back(refRank - srcRank + i);
            }
        }
        std::string bcDims;
        for (size_t i = 0; i < bcDimsVec.size(); ++i) {
            if (i > 0) bcDims += ", ";
            bcDims += std::to_string(bcDimsVec[i]);
        }

        // Classify source dims: static >1 or leading Elipsis → known_nonexpanding,
        // static ==1 → known_expanding. Trailing Elipsis left unclassified.
        std::string knownExp, knownNonExp;
        if (const auto* srcTTk = std::get_if<mdl::TensorType>(&srcDT)) {
            for (int i = 0; i < srcRank; ++i) {
                const auto& dim = srcTTk->shape[i];
                if (std::holds_alternative<mdl::Elipsis>(dim)) {
                    if (leadingAlignment) {
                        if (!knownNonExp.empty()) knownNonExp += ", ";
                        knownNonExp += std::to_string(i);
                    }
                } else {
                    int64_t val = -2;
                    if (const auto* u = std::get_if<unsigned>(&dim)) val = *u;
                    else if (const auto* n = std::get_if<const mdl::Node*>(&dim))
                        val = tryEvalConst(**n, argVals).value_or(-2);
                    if (val == 1) {
                        if (!knownExp.empty()) knownExp += ", ";
                        knownExp += std::to_string(i);
                    } else if (val > 1) {
                        if (!knownNonExp.empty()) knownNonExp += ", ";
                        knownNonExp += std::to_string(i);
                    }
                }
            }
        }
        std::string knownExpAttr    = knownExp.empty()    ? "array<i64>" : ("array<i64: " + knownExp    + ">");
        std::string knownNonExpAttr = knownNonExp.empty() ? "array<i64>" : ("array<i64: " + knownNonExp + ">");

        bool hasDyn = std::any_of(refTT->shape.begin(), refTT->shape.end(),
            [](const auto& d) { return std::holds_alternative<mdl::Elipsis>(d); });

        if (!hasDyn)
            return bindOp(std::format(
                "stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
                srcSSA, bcDims, srcMlir, refMlir), "bc");

        // Dynamic: build runtime shape tensor from refSSA
        std::vector<std::string> parts;
        for (size_t i = 0; i < refTT->shape.size(); ++i) {
            std::visit(mdl::overloaded{
                [&](mdl::Elipsis) {
                    auto s32 = bindOp(std::format(
                        "stablehlo.get_dimension_size {}, dim = {} : ({}) -> tensor<i32>",
                        refSSA, i, refMlir), "bda");
                    auto s64 = bindOp(std::format(
                        "stablehlo.convert {} : (tensor<i32>) -> tensor<i64>", s32), "bdb");
                    parts.push_back(bindOp(std::format(
                        "stablehlo.reshape {} : (tensor<i64>) -> tensor<1xi64>", s64), "bdc"));
                },
                [&](unsigned u) {
                    parts.push_back(bindOp(std::format(
                        "stablehlo.constant dense<{}> : tensor<1xi64>", u), "bdk"));
                },
                [&](const mdl::Node* n) {
                    parts.push_back(bindOp(std::format(
                        "stablehlo.constant dense<{}> : tensor<1xi64>",
                        tryEvalConst(*n, argVals).value_or(1)), "bdk"));
                },
            }, refTT->shape[i]);
        }

        std::string shapeSSA;
        if (parts.size() == 1) {
            shapeSSA = parts[0];
        } else {
            std::string ca, ct;
            for (size_t k = 0; k < parts.size(); ++k) {
                if (k > 0) { ca += ", "; ct += ", "; }
                ca += parts[k]; ct += "tensor<1xi64>";
            }
            shapeSSA = bindOp(std::format(
                "stablehlo.concatenate {}, dim = 0 : ({}) -> tensor<{}xi64>",
                ca, ct, parts.size()), "bshp");
        }

        // array<i64> requires no colon when empty; "array<i64: >" is invalid syntax
        std::string bcDimsAttr = bcDims.empty() ? "array<i64>" : ("array<i64: " + bcDims + ">");
        return bindOp(std::format(
            "\"stablehlo.dynamic_broadcast_in_dim\"({}, {}) {{"
            "broadcast_dimensions = {}, "
            "known_expanding_dimensions = {}, "
            "known_nonexpanding_dimensions = {}"
            "}} : ({}, tensor<{}xi64>) -> {}",
            srcSSA, shapeSSA, bcDimsAttr, knownExpAttr, knownNonExpAttr,
            srcMlir, parts.size(), refMlir), "bc");
    };

    // ── output type with last axis reduced (keepdims or not) ──────────────────
    auto reduceOutType = [&](const mdl::DataType& dt, bool keepdims) -> std::string {
        const auto* tt = std::get_if<mdl::TensorType>(&dt);
        assert(tt && !tt->shape.empty());
        std::string s = "tensor<";
        for (size_t i = 0; i + 1 < tt->shape.size(); ++i) {
            if (i > 0) s += "x";
            s += dimStr(tt->shape[i], argVals);
        }
        if (keepdims) {
            if (tt->shape.size() > 1) s += "x";
            s += "1";
        }
        s += "x" + elemTypeStr(tt->type) + ">";
        return s;
    };

    // ── extract last-dim float count from a TensorType ────────────────────────
    auto lastDimF = [&](const mdl::TensorType& tt) -> float {
        if (tt.shape.empty()) return 1.0f;
        const auto& back = tt.shape.back();
        if (const unsigned* u = std::get_if<unsigned>(&back))
            return static_cast<float>(*u);
        if (const mdl::Node* const* np = std::get_if<const mdl::Node*>(&back))
            if (auto v = tryEvalConst(**np, argVals))
                return static_cast<float>(*v);
        return 1.0f;
    };

    // ── recursive node emitter ─────────────────────────────────────────────────
    std::function<std::string(const mdl::Node&)> emitNode;
    emitNode = [&](const mdl::Node& node) -> std::string {
        if (auto it = cache.find(&node); it != cache.end())
            return it->second;

        std::string result = std::visit(mdl::overloaded{

            [&](const mdl::ConstNode& cn) -> std::string {
                return std::visit(mdl::overloaded{
                    // Integer literals in MDL arithmetic are always float context.
                    [&](int v) {
                        return bindOp(std::format(
                            "stablehlo.constant dense<{:.8e}> : tensor<f32>",
                            static_cast<float>(v)), "c");
                    },
                    [&](double v) {
                        return bindOp(std::format(
                            "stablehlo.constant dense<{:.8e}> : tensor<f32>",
                            static_cast<float>(v)), "c");
                    },
                    // vector<int>: consumed by the PERMUTE parent, not emitted standalone
                    [](const std::vector<int>&) -> std::string { return "__vec__"; },
                }, cn.value);
            },

            [](const mdl::ParameterNode&) -> std::string {
                assert(false && "ParameterNode must be pre-seeded in cache");
                return "";
            },

            [&](const mdl::HyperparameterNode& hn) -> std::string {
                // Hyperparameters are scalar integers used in float expressions.
                if (auto v = argVals.lookup(hn.name))
                    return bindOp(std::format(
                        "stablehlo.constant dense<{:.8e}> : tensor<f32>",
                        static_cast<float>(*v)), "hp");
                return bindOp("stablehlo.constant dense<0.0e+00> : tensor<f32>", "hp");
            },

            [](const mdl::ErrorNode&) -> std::string {
                std::abort(); // should never compile an ErrorNode
            },

            [&](const mdl::InvocationNode& inv) -> std::string {
                using FIDs = mdl::builtins::FunctionIDs;

                auto arg = [&](size_t i) { return emitNode(inv.arguments[i].get()); };
                const std::string outType = mlirTypeStr(inv.type, argVals);
                const mdl::DataType& lhsDT = mdl::getDataType(inv.arguments[0].get());
                const std::string inType   = mlirTypeStr(lhsDT, argVals);
                const std::string et = std::visit(mdl::overloaded{
                    [](mdl::PrimitiveType pt)      { return elemTypeStr(pt); },
                    [](const mdl::TensorType& tt)  { return elemTypeStr(tt.type); },
                    [](const auto&) -> std::string { return "f32"; },
                }, lhsDT);

                // True if srcDT needs broadcast to become shape-compatible with refDT.
                auto needsBc = [](const mdl::DataType& src, const mdl::DataType& ref) -> bool {
                    int sr = rankOf(src), rr = rankOf(ref);
                    if (sr < rr) return true;
                    if (sr > rr) return false;
                    const auto* s = std::get_if<mdl::TensorType>(&src);
                    const auto* r = std::get_if<mdl::TensorType>(&ref);
                    if (!s || !r) return false;
                    for (size_t k = 0; k < s->shape.size() && k < r->shape.size(); ++k) {
                        const auto* su = std::get_if<unsigned>(&s->shape[k]);
                        if (!su || *su != 1) continue;
                        const auto* du = std::get_if<unsigned>(&r->shape[k]);
                        if (!du || *du != 1) return true;
                    }
                    return false;
                };

                switch (inv.id) {

                // ── element-wise binary ──────────────────────────────────────
                case FIDs::TENSOR_ADD: {
                    const mdl::DataType& rhsDT = mdl::getDataType(inv.arguments[1].get());
                    const std::string lhs0 = arg(0), rhs0 = arg(1);
                    const std::string lhsSSA = needsBc(lhsDT, rhsDT)
                        ? broadcastTo(lhs0, lhsDT, rhs0, rhsDT) : lhs0;
                    const std::string rhsSSA = needsBc(rhsDT, lhsDT)
                        ? broadcastTo(rhs0, rhsDT, lhs0, lhsDT) : rhs0;
                    return bindOp(std::format("stablehlo.add {}, {} : {}", lhsSSA, rhsSSA, outType));
                }
                case FIDs::TENSOR_SUB: {
                    const mdl::DataType& rhsDT = mdl::getDataType(inv.arguments[1].get());
                    const std::string lhs0 = arg(0), rhs0 = arg(1);
                    const std::string lhsSSA = needsBc(lhsDT, rhsDT)
                        ? broadcastTo(lhs0, lhsDT, rhs0, rhsDT) : lhs0;
                    const std::string rhsSSA = needsBc(rhsDT, lhsDT)
                        ? broadcastTo(rhs0, rhsDT, lhs0, lhsDT) : rhs0;
                    return bindOp(std::format("stablehlo.subtract {}, {} : {}", lhsSSA, rhsSSA, outType));
                }
                case FIDs::TENSOR_PROD: {
                    const mdl::DataType& rhsDT = mdl::getDataType(inv.arguments[1].get());
                    const std::string lhs0 = arg(0), rhs0 = arg(1);
                    const std::string lhsSSA = needsBc(lhsDT, rhsDT)
                        ? broadcastTo(lhs0, lhsDT, rhs0, rhsDT) : lhs0;
                    const std::string rhsSSA = needsBc(rhsDT, lhsDT)
                        ? broadcastTo(rhs0, rhsDT, lhs0, lhsDT) : rhs0;
                    return bindOp(std::format("stablehlo.multiply {}, {} : {}", lhsSSA, rhsSSA, outType));
                }
                case FIDs::TENSOR_DIV: {
                    const mdl::DataType& rhsDT = mdl::getDataType(inv.arguments[1].get());
                    const std::string lhs0 = arg(0), rhs0 = arg(1);
                    const std::string lhsSSA = needsBc(lhsDT, rhsDT)
                        ? broadcastTo(lhs0, lhsDT, rhs0, rhsDT) : lhs0;
                    const std::string rhsSSA = needsBc(rhsDT, lhsDT)
                        ? broadcastTo(rhs0, rhsDT, lhs0, lhsDT) : rhs0;
                    return bindOp(std::format("stablehlo.divide {}, {} : {}", lhsSSA, rhsSSA, outType));
                }

                // ── matmul ────────────────────────────────────────────────────
                case FIDs::TENSOR_DOT: {
                    const int rA = rankOf(lhsDT);
                    const int rB = rankOf(mdl::getDataType(inv.arguments[1].get()));
                    const std::string rhsType =
                        mlirTypeStr(mdl::getDataType(inv.arguments[1].get()), argVals);
                    if (rA <= 2 && rB <= 2)
                        return bindOp(std::format(
                            "stablehlo.dot {}, {} : ({}, {}) -> {}",
                            arg(0), arg(1), inType, rhsType, outType));
                    // Batched: all leading dims are batch dims
                    int nBatch = rA - 2;
                    std::string bds;
                    for (int b = 0; b < nBatch; ++b) {
                        if (b > 0) bds += ", ";
                        bds += std::to_string(b);
                    }
                    return bindOp(std::format(
                        "stablehlo.dot_general {}, {},"
                        " batching_dims = [{}] x [{}],"
                        " contracting_dims = [{}] x [{}]"
                        " : ({}, {}) -> {}",
                        arg(0), arg(1), bds, bds, rA - 1, rB - 2,
                        inType, rhsType, outType), "mm");
                }

                // ── reductions ────────────────────────────────────────────────
                case FIDs::MAX:
                    return emitReduce(arg(0), inType, outType, et,
                        "-3.40282347E+38", "stablehlo.maximum", rankOf(lhsDT) - 1);
                case FIDs::MIN:
                    return emitReduce(arg(0), inType, outType, et,
                        "3.40282347E+38", "stablehlo.minimum", rankOf(lhsDT) - 1);
                case FIDs::SUM:
                    return emitReduce(arg(0), inType, outType, et,
                        "0.0", "stablehlo.add", rankOf(lhsDT) - 1);

                case FIDs::MEAN: {
                    const auto* tt = std::get_if<mdl::TensorType>(&lhsDT);
                    assert(tt);
                    int axis = static_cast<int>(tt->shape.size()) - 1;
                    mdl::TensorType sumTT = *tt;
                    if (!sumTT.shape.empty()) sumTT.shape.pop_back();
                    mdl::DataType sumDT = sumTT;
                    std::string sumType = reduceOutType(lhsDT, /*keepdims=*/false);
                    std::string sumSSA  = emitReduce(arg(0), inType, sumType, et,
                        "0.0", "stablehlo.add", axis);
                    float invN = 1.0f / std::max(lastDimF(*tt), 1.0f);
                    std::string sc = bindOp(std::format(
                        "stablehlo.constant dense<{:.8e}> : tensor<f32>", invN), "invn");
                    std::string scBc = broadcastTo(sc, mdl::PrimitiveType::floating, sumSSA, sumDT);
                    return bindOp(std::format("stablehlo.multiply {}, {} : {}", sumSSA, scBc, sumType));
                }

                case FIDs::VAR: {
                    const auto* tt = std::get_if<mdl::TensorType>(&lhsDT);
                    assert(tt);
                    int axis = static_cast<int>(tt->shape.size()) - 1;
                    mdl::TensorType sumTT = *tt;
                    if (!sumTT.shape.empty()) sumTT.shape.pop_back();
                    mdl::DataType sumDT = sumTT;
                    std::string sumType = reduceOutType(lhsDT, /*keepdims=*/false);
                    float invN = 1.0f / std::max(lastDimF(*tt), 1.0f);
                    std::string sc = bindOp(std::format(
                        "stablehlo.constant dense<{:.8e}> : tensor<f32>", invN), "invn");
                    // sum(x), E[x]
                    std::string sx  = emitReduce(arg(0), inType, sumType, et,
                        "0.0", "stablehlo.add", axis);
                    std::string scBc = broadcastTo(sc, mdl::PrimitiveType::floating, sx, sumDT);
                    std::string ex  = bindOp(std::format("stablehlo.multiply {}, {} : {}",
                        sx, scBc, sumType), "ex");
                    // sum(x²), E[x²]
                    std::string x2  = bindOp(std::format("stablehlo.multiply {}, {} : {}",
                        arg(0), arg(0), inType), "sq");
                    std::string sx2 = emitReduce(x2, inType, sumType, et,
                        "0.0", "stablehlo.add", axis);
                    std::string ex2 = bindOp(std::format("stablehlo.multiply {}, {} : {}",
                        sx2, scBc, sumType), "ex2");
                    // var = E[x²] − E[x]²
                    std::string exsq  = bindOp(std::format("stablehlo.multiply {}, {} : {}",
                        ex, ex, sumType), "exsq");
                    return bindOp(std::format("stablehlo.subtract {}, {} : {}",
                        ex2, exsq, sumType), "var");
                }

                // ── unary ─────────────────────────────────────────────────────
                case FIDs::COS:
                    return bindOp(std::format("stablehlo.cosine {} : {}", arg(0), outType));
                case FIDs::SIN:
                    return bindOp(std::format("stablehlo.sine {} : {}", arg(0), outType));
                case FIDs::EXP:
                    return bindOp(std::format("stablehlo.exponential {} : {}", arg(0), outType));
                case FIDs::SQRT:
                    return bindOp(std::format("stablehlo.sqrt {} : {}", arg(0), outType));
                case FIDs::RSQRT:
                    return bindOp(std::format("stablehlo.rsqrt {} : {}", arg(0), outType));
                case FIDs::ERF:
                    // chlo.erf uses stablehlo-style "intype -> outtype" syntax
                    return bindOp(std::format("chlo.erf {} : {} -> {}", arg(0), inType, outType));
                case FIDs::TAN: {
                    std::string s = bindOp(std::format("stablehlo.sine {} : {}",
                        arg(0), outType), "sin");
                    std::string c = bindOp(std::format("stablehlo.cosine {} : {}",
                        arg(0), outType), "cos");
                    return bindOp(std::format("stablehlo.divide {}, {} : {}",
                        s, c, outType), "tan");
                }

                // ── shape ops ─────────────────────────────────────────────────
                case FIDs::RESHAPE: {
                    const auto* outTT = std::get_if<mdl::TensorType>(&inv.type);
                    assert(outTT);

                    bool hasDynamic = std::any_of(outTT->shape.begin(), outTT->shape.end(),
                        [](const auto& d) { return std::holds_alternative<mdl::Elipsis>(d); });

                    std::string inputSSA = arg(0);

                    if (!hasDynamic)
                        return bindOp(std::format("stablehlo.reshape {} : ({}) -> {}",
                            inputSSA, inType, outType), "rs");

                    // ── dynamic reshape ──────────────────────────────────────
                    // inv.arguments: [0]=input, [1..outRank]=shape spec sentinels
                    //   sentinel -1 → passthrough (copy Elipsis dim from input)
                    //   sentinel -2 → infer (product of remaining input dims / known output dims)
                    //   positive N  → static constant N
                    const auto* inTT = std::get_if<mdl::TensorType>(&lhsDT);
                    assert(inTT);
                    const int outRank = static_cast<int>(outTT->shape.size());

                    // stablehlo.dynamic_reshape is illegal in IREE; use tensor.expand_shape
                    // (split a dim) or tensor.collapse_shape (merge dims) instead.

                    // Locate Elipsis (dynamic) dims in input — consumed by -1 passthroughs in order
                    std::vector<int> inElipsisDims;
                    for (int i = 0; i < static_cast<int>(inTT->shape.size()); ++i)
                        if (std::holds_alternative<mdl::Elipsis>(inTT->shape[i]))
                            inElipsisDims.push_back(i);

                    // Count passthroughs (-1) and accumulate product of static output dims
                    int numPassthrough = 0;
                    int64_t staticOutKnown = 1;
                    for (int i = 0; i < outRank; ++i) {
                        auto v = tryEvalConst(inv.arguments[i + 1].get(), argVals);
                        if (v && *v == -1) numPassthrough++;
                        else if (v && *v > 0) staticOutKnown *= *v;
                    }

                    // Compute staticRemProd: product of non-passthrough input static dims
                    int64_t staticRemProd = 1;
                    bool remHasDyn = false;
                    {
                        int elUsed = 0;
                        for (int i = 0; i < static_cast<int>(inTT->shape.size()); ++i) {
                            if (std::holds_alternative<mdl::Elipsis>(inTT->shape[i])
                                    && elUsed < numPassthrough) { ++elUsed; continue; }
                            auto dv = std::visit(mdl::overloaded{
                                [](unsigned u)          -> std::optional<int64_t> { return static_cast<int64_t>(u); },
                                [&](const mdl::Node* n) -> std::optional<int64_t> { return tryEvalConst(*n, argVals); },
                                [](mdl::Elipsis)        -> std::optional<int64_t> { return std::nullopt; },
                            }, inTT->shape[i]);
                            if (dv) staticRemProd *= *dv; else remHasDyn = true;
                        }
                    }

                    // Compute per-output-dim actual values (more precise than MDL type inference
                    // which sets infer dims to Elipsis even when they're statically known)
                    std::vector<std::optional<int64_t>> actualOut(outRank);
                    {
                        int64_t inferVal = (!remHasDyn && staticOutKnown > 0)
                            ? staticRemProd / staticOutKnown : 0;
                        int pti = 0;
                        for (int i = 0; i < outRank && i + 1 < static_cast<int>(inv.arguments.size()); ++i) {
                            auto v = tryEvalConst(inv.arguments[i + 1].get(), argVals);
                            if (!v)                  { actualOut[i] = std::nullopt; }
                            else if (*v == -1)       { actualOut[i] = std::nullopt; ++pti; }
                            else if (*v == -2)       { actualOut[i] = inferVal > 0 ? std::make_optional(inferVal) : std::nullopt; }
                            else if (*v > 0)         { actualOut[i] = *v; }
                            else                     { actualOut[i] = std::nullopt; }
                        }
                    }

                    // Build the tighter MLIR output type (uses static value for infer dim when known)
                    std::string actualOutType = "tensor<";
                    for (int i = 0; i < outRank; ++i) {
                        if (i > 0) actualOutType += "x";
                        actualOutType += actualOut[i] ? std::to_string(*actualOut[i]) : "?";
                    }
                    actualOutType += "xf32>";

                    int inRankI = static_cast<int>(inTT->shape.size());
                    int inRemaining  = inRankI - numPassthrough;
                    int outRemaining = outRank  - numPassthrough;

                    // Build reassociation list for tensor.expand_shape / tensor.collapse_shape.
                    // nSingle: number of leading single-element groups [[0], [1], ...]
                    // lastStart: index where the final multi-element group begins
                    // lastSize: how many dims in that final group
                    auto buildReassoc = [&](int nSingle, int lastStart, int lastSize) -> std::string {
                        std::string s = "[";
                        for (int i = 0; i < nSingle; ++i) {
                            if (i > 0) s += ", ";
                            s += "[" + std::to_string(i) + "]";
                        }
                        if (nSingle > 0 && lastSize > 0) s += ", ";
                        if (lastSize > 0) {
                            s += "[";
                            for (int j = 0; j < lastSize; ++j) {
                                if (j > 0) s += ", ";
                                s += std::to_string(lastStart + j);
                            }
                            s += "]";
                        }
                        s += "]";
                        return s;
                    };

                    if (outRemaining > inRemaining) {
                        // EXPAND: split (inRemaining==1) or unsqueeze (inRemaining==0)
                        // When inRemaining==0, the last passthrough input dim absorbs the extra
                        // output dims; when inRemaining==1, a dedicated non-passthrough dim splits.
                        int nSingle    = (inRemaining == 0) ? (numPassthrough - 1) : numPassthrough;
                        int lastStart  = nSingle;
                        int lastSize   = outRank - nSingle;
                        std::string reassoc = buildReassoc(nSingle, lastStart, lastSize);

                        // output_shape: index SSAs for dynamic dims, integer literals for static
                        std::string outputShape;
                        int pti = 0;
                        for (int i = 0; i < outRank; ++i) {
                            if (i > 0) outputShape += ", ";
                            if (actualOut[i]) {
                                outputShape += std::to_string(*actualOut[i]);
                            } else {
                                int inDim = (pti < static_cast<int>(inElipsisDims.size()))
                                    ? inElipsisDims[pti] : pti;
                                ++pti;
                                std::string ci = bindOp(
                                    std::format("arith.constant {} : index", inDim), "ci");
                                outputShape += bindOp(
                                    std::format("tensor.dim {}, {} : {}", inputSSA, ci, inType), "td");
                            }
                        }

                        std::string expanded = bindOp(std::format(
                            "tensor.expand_shape {} {} output_shape [{}] : {} into {}",
                            inputSSA, reassoc, outputShape, inType, actualOutType), "rs");
                        if (actualOutType == outType) return expanded;
                        return bindOp(std::format(
                            "tensor.cast {} : {} to {}", expanded, actualOutType, outType), "rs");

                    } else if (outRemaining < inRemaining) {
                        // COLLAPSE: merge (outRemaining==1) or squeeze (outRemaining==0)
                        int nSingle   = (outRemaining == 0) ? (numPassthrough - 1) : numPassthrough;
                        int lastStart = nSingle;
                        int lastSize  = inRankI - nSingle;
                        std::string reassoc = buildReassoc(nSingle, lastStart, lastSize);

                        std::string collapsed = bindOp(std::format(
                            "tensor.collapse_shape {} {} : {} into {}",
                            inputSSA, reassoc, inType, actualOutType), "rs");
                        if (actualOutType == outType) return collapsed;
                        return bindOp(std::format(
                            "tensor.cast {} : {} to {}", collapsed, actualOutType, outType), "rs");

                    } else {
                        // Same remaining count: pure type annotation change
                        if (inType == outType) return inputSSA;
                        return bindOp(std::format(
                            "tensor.cast {} : {} to {}", inputSSA, inType, outType), "rs");
                    }
                }

                case FIDs::PERMUTE: {
                    const auto* cn =
                        std::get_if<mdl::ConstNode>(&inv.arguments[1].get());
                    assert(cn && std::holds_alternative<std::vector<int>>(cn->value));
                    const auto& perm = std::get<std::vector<int>>(cn->value);
                    int rank = rankOf(lhsDT);
                    // Prepend leading identity indices for partial permutations
                    int offset = (rank > 0) ? (rank - static_cast<int>(perm.size())) : 0;
                    std::string dims;
                    for (int i = 0; i < offset; ++i) {
                        if (!dims.empty()) dims += ", ";
                        dims += std::to_string(i);
                    }
                    for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
                        if (!dims.empty()) dims += ", ";
                        int idx = perm[i];
                        if (idx < 0 && rank > 0) idx += rank;
                        dims += std::to_string(idx);
                    }
                    return bindOp(std::format(
                        "stablehlo.transpose {}, dims = [{}] : ({}) -> {}",
                        arg(0), dims, inType, outType), "tp");
                }

                case FIDs::EXPAND: {
                    int rankIn  = rankOf(lhsDT);
                    int rankOut = rankOf(inv.type);
                    const auto* outTT = std::get_if<mdl::TensorType>(&inv.type);
                    const auto* inTT  = std::get_if<mdl::TensorType>(&lhsDT);

                    // broadcast_dimensions: trailing alignment
                    std::string dims;
                    for (int i = 0; i < rankIn; ++i) {
                        if (i > 0) dims += ", ";
                        dims += std::to_string(rankOut - rankIn + i);
                    }

                    bool hasDyn = outTT && std::any_of(outTT->shape.begin(), outTT->shape.end(),
                        [](const auto& d) { return std::holds_alternative<mdl::Elipsis>(d); });

                    if (!hasDyn)
                        return bindOp(std::format(
                            "stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
                            arg(0), dims, inType, outType), "bc");

                    // Dynamic: build shape tensor.
                    // New leading dims (offset = rankOut-rankIn) from inv.type; must be static.
                    // Input-mapped trailing dims read from arg(0) at runtime.
                    std::string inputSSA = arg(0);
                    int offset = rankOut - rankIn;
                    std::vector<std::string> parts;
                    for (int i = 0; i < rankOut; ++i) {
                        auto emitConst = [&](int64_t v) {
                            parts.push_back(bindOp(std::format(
                                "stablehlo.constant dense<{}> : tensor<1xi64>", v), "bdk"));
                        };
                        if (i < offset) {
                            std::visit(mdl::overloaded{
                                [&](unsigned u)          { emitConst(u); },
                                [&](const mdl::Node* n)  { emitConst(tryEvalConst(*n, argVals).value_or(1)); },
                                [&](mdl::Elipsis)        { emitConst(1); },
                            }, outTT->shape[i]);
                        } else {
                            int j = i - offset;
                            if (inTT && j < static_cast<int>(inTT->shape.size())) {
                                std::visit(mdl::overloaded{
                                    [&](mdl::Elipsis) {
                                        auto s32 = bindOp(std::format(
                                            "stablehlo.get_dimension_size {}, dim = {} : ({}) -> tensor<i32>",
                                            inputSSA, j, inType), "bda");
                                        auto s64 = bindOp(std::format(
                                            "stablehlo.convert {} : (tensor<i32>) -> tensor<i64>", s32), "bdb");
                                        parts.push_back(bindOp(std::format(
                                            "stablehlo.reshape {} : (tensor<i64>) -> tensor<1xi64>", s64), "bdc"));
                                    },
                                    [&](unsigned u)         { emitConst(u); },
                                    [&](const mdl::Node* n) { emitConst(tryEvalConst(*n, argVals).value_or(1)); },
                                }, inTT->shape[j]);
                            } else {
                                emitConst(1);
                            }
                        }
                    }

                    std::string shapeSSA;
                    if (parts.size() == 1) {
                        shapeSSA = parts[0];
                    } else {
                        std::string ca, ct;
                        for (size_t k = 0; k < parts.size(); ++k) {
                            if (k > 0) { ca += ", "; ct += ", "; }
                            ca += parts[k]; ct += "tensor<1xi64>";
                        }
                        shapeSSA = bindOp(std::format(
                            "stablehlo.concatenate {}, dim = 0 : ({}) -> tensor<{}xi64>",
                            ca, ct, parts.size()), "bshp");
                    }

                    std::string dimsAttr = dims.empty() ? "array<i64>" : ("array<i64: " + dims + ">");
                    return bindOp(std::format(
                        "\"stablehlo.dynamic_broadcast_in_dim\"({}, {}) {{"
                        "broadcast_dimensions = {}, "
                        "known_expanding_dimensions = array<i64>, "
                        "known_nonexpanding_dimensions = array<i64>"
                        "}} : ({}, tensor<{}xi64>) -> {}",
                        inputSSA, shapeSSA, dimsAttr, inType, parts.size(), outType), "bc");
                }

                default:
                    assert(false && "unhandled FunctionID in IREE backend");
                    return "";
                }
            },

        }, node);

        cache[&node] = result;
        return result;
    };

    // ── Build function text ────────────────────────────────────────────────────
    std::ostringstream func_text;
    func_text << "  func.func @" << mdl::mangledName(moduleName, func.name) << "(";
    for (size_t i = 0; i < paramNodes.size(); ++i) {
        if (i > 0) func_text << ", ";
        const auto& pn = std::get<mdl::ParameterNode>(*paramNodes[i]);
        func_text << "%arg" << i << ": " << mlirTypeStr(mdl::getDataType(pn), argVals);
        logger->info("  arg[{}] = %arg{} : {}", i, i,
            mlirTypeStr(mdl::getDataType(pn), argVals));
    }
    const std::string retType = mlirTypeStr(mdl::getDataType(func.output), argVals);
    func_text << ") -> " << retType << " {\n";

    std::string resultSSA = emitNode(func.output);

    func_text << body.str();
    func_text << "    return " << resultSSA << " : " << retType << "\n";
    func_text << "  }\n";

    return func_text.str();
}

// ── Top-level entry point ─────────────────────────────────────────────────────

void mdl::backends::iree::compileToFile(
    const mdl::CompilationUnit& unit,
    const std::filesystem::path& output,
    const mdl::ArgVals& argVals,
    const mdl::CompileOptions& /*opts*/)
{
    auto logger = mdl::log::getLogger("iree-backend");

    std::filesystem::path vmfbPath = output;
    vmfbPath.replace_extension(".vmfb");

    // ── Phase 1: emit StableHLO MLIR ─────────────────────────────────────────
    std::ostringstream mlirOut;
    mlirOut << "// Generated by mdlc (IREE backend)\n";
    mlirOut << "module {\n\n";

    for (const auto& [name, member] : unit.definitions) {
        const mdl::ModuleType* mod = std::get_if<mdl::ModuleType>(&member);
        if (!mod) continue;
        // Skip modules whose scalar typeargs aren't in argVals.
        // Struct-typed typeargs (InvalidDataType) are OK — their leaf fields are in argVals.
        bool hasUnresolvable = false;
        for (const auto& [tname, hn] : mod->typeargs) {
            auto res = argVals.lookup(hn.get().name);
            if (!res.has_value() && res.error() == mdl::ArgVals::ErrorCode::NotFound) {
                hasUnresolvable = true;
                break;
            }
        }
        if (hasUnresolvable) {
            logger->warn("Skipping module {} (unresolvable type args)", name);
            continue;
        }
        for (const auto& [fname, funcDecl] : mod->functions) {
            if (!funcDecl.templateParameters.empty()) continue;
            mdl::FunctionDecl mangledDecl = funcDecl;
            mangledDecl.name = mdl::mangledName(name, fname);
            mlirOut << translateFuncToMlir(mangledDecl, argVals, name) << "\n";
        }
    }

    mlirOut << "} // end module\n";
    const std::string mlirText = mlirOut.str();
    logger->info("Generated StableHLO MLIR ({} bytes)", mlirText.size());
    logger->debug("MLIR:\n{}", mlirText);

    // ── Phase 2: compile via iree-compile subprocess ──────────────────────────
    // libIREECompiler.so embeds LLVM 20 while TVM links system LLVM 19 — loading
    // both in-process causes a symbol collision and segfault.  Running iree-compile
    // as a child process avoids the conflict entirely.

    namespace fs = std::filesystem;
    const fs::path mlirTmp = fs::temp_directory_path() / "mdl_iree_module.mlir";

    {
        std::ofstream f(mlirTmp);
        if (!f) {
            logger->error("IREE: could not write temp MLIR to {}", mlirTmp.string());
            return;
        }
        f << mlirText;
    }
    logger->info("[mdl] Wrote temp MLIR to {}", mlirTmp.string());

    // Locate iree-compile: honour IREE_COMPILE env, then PATH.
    std::string ireeCompileBin = "iree-compile";
    if (const char* env = std::getenv("IREE_COMPILE")) ireeCompileBin = env;

    const std::string cmd =
        ireeCompileBin +
        " --iree-input-type=stablehlo"
        " --iree-hal-target-backends=llvm-cpu"
        " --iree-llvmcpu-target-cpu=host"
        " --iree-llvmcpu-target-cpu-features=host"
        " --iree-llvmcpu-enable-ukernels=all"
        " --mlir-print-debuginfo=false"
        " -o " + vmfbPath.string() +
        " " + mlirTmp.string() +
        " 2>&1";

    logger->info("[mdl] Running: {}", cmd);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        logger->error("IREE: iree-compile exited with code {}", rc);
        return;
    }
    logger->info("[mdl] IREE: wrote {}", vmfbPath.string());
}
