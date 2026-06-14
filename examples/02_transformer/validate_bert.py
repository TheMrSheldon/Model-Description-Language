"""
Validates TransformerEncoder__forward against HuggingFace BERT's first encoder layer.

Weights and reference output are taken directly from bert-base-uncased layer 0.
The MDL model matches BERT's architecture:
  - Multi-head self-attention (12 heads, dim_key=64, dim_value=64)
  - Attention output projection + residual + LayerNorm
  - FFN (768 -> 3072 -> 768) with GELU activation + residual + LayerNorm

Usage:
    python validate_bert.py [path/to/transformer_out.so]

The .so defaults to /tmp/transformer_out.so if not provided.  Compile it with:
    ulimit -s unlimited && mdlc examples/02_transformer/model.dlm \\
        -a examples/02_transformer/model.yml -o /tmp/transformer_out.so -vv
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

SO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/transformer_out.so"

# ---------------------------------------------------------------------------
# 1. Load HuggingFace BERT and extract layer-0 weights
# ---------------------------------------------------------------------------
print("Loading bert-base-uncased ...")
config = BertConfig()  # standard bert-base: hidden=768, heads=12, intermediate=3072
model = BertModel(config)
model.eval()

layer = model.encoder.layer[0]

# Attention weights  (all [768, 768])
wQ = layer.attention.self.query.weight.detach().float()   # [768, 768]
bQ = layer.attention.self.query.bias.detach().float()     # [768]
wK = layer.attention.self.key.weight.detach().float()
bK = layer.attention.self.key.bias.detach().float()
wV = layer.attention.self.value.weight.detach().float()
bV = layer.attention.self.value.bias.detach().float()
wO = layer.attention.output.dense.weight.detach().float() # [768, 768]
bO = layer.attention.output.dense.bias.detach().float()
mha_norm_gamma = layer.attention.output.LayerNorm.weight.detach().float()  # [768]
mha_norm_beta  = layer.attention.output.LayerNorm.bias.detach().float()

# FFN weights
wI  = layer.intermediate.dense.weight.detach().float()    # [3072, 768]
bI  = layer.intermediate.dense.bias.detach().float()      # [3072]
wIO = layer.output.dense.weight.detach().float()          # [768, 3072]
bIO = layer.output.dense.bias.detach().float()            # [768]
ffn_norm_gamma = layer.output.LayerNorm.weight.detach().float()  # [768]
ffn_norm_beta  = layer.output.LayerNorm.bias.detach().float()

# ---------------------------------------------------------------------------
# 2. Create a small random input and compute the HF reference output
# ---------------------------------------------------------------------------
BATCH, SEQ, DIM = 1, 8, 768
torch.manual_seed(42)
x_torch = torch.randn(BATCH, SEQ, DIM)

# BertLayer expects the extended attention mask (additive, float).
# All-zero mask means no masking — full self-attention over all positions.
attn_mask_extended = torch.zeros(BATCH, 1, 1, SEQ)

with torch.no_grad():
    hf_output = layer(x_torch, attention_mask=attn_mask_extended)[0]  # [B, T, 768]

print(f"HF output shape: {hf_output.shape}, mean={hf_output.mean():.6f}")

# ---------------------------------------------------------------------------
# 3. Call the MDL-compiled TransformerEncoder__forward
#
# Argument order (DFS traversal of the compute graph, confirmed by compiler manifest):
#   [0]  x                   float[B, T, 768]
#   [1]  weightQuery.W       float[768, 768]
#   [2]  weightQuery.b       float[768]
#   [3]  weightKey.W         float[768, 768]
#   [4]  weightKey.b         float[768]
#   [5]  weightValue.W       float[768, 768]
#   [6]  weightValue.b       float[768]
#   [7]  weightOut.W         float[768, 768]
#   [8]  weightOut.b         float[768]
#   [9]  mhaNorm.gamma       float[768]
#   [10] mhaNorm.beta        float[768]
#   [11] weightIntermed.W    float[3072, 768]
#   [12] weightIntermed.b    float[3072]
#   [13] weightIntermedOut.W float[768, 3072]
#   [14] weightIntermedOut.b float[768]
#   [15] ffnNorm.gamma       float[768]
#   [16] ffnNorm.beta        float[768]
#   [17] output (written in-place)
# ---------------------------------------------------------------------------
if not os.path.exists(SO_PATH):
    print(f"\nSkipping MDL call: {SO_PATH} not found.")
    sys.exit(0)

# Set up TVM Python path from the in-tree TVM build
_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_tvm_py = os.path.join(_repo, "out/build/conf-dev/_deps/tvm-src/python")
_tvm_lib = os.path.join(_repo, "out/build/conf-dev/lib")
if os.path.isdir(_tvm_py):
    sys.path.insert(0, _tvm_py)
    os.environ["LD_LIBRARY_PATH"] = _tvm_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Use physical cores only — hyperthreads add pthread barrier overhead without compute gain
# at the small batch sizes we benchmark. Must be set before tvm.runtime is imported.
os.environ.setdefault("TVM_NUM_THREADS", str(max(1, os.cpu_count() // 2)))

import tvm.runtime as R
import tvm.runtime.vm

print(f"\nLoading {SO_PATH} ...")
exec_mod = R.load_module(SO_PATH)
vm = tvm.runtime.vm.VirtualMachine(exec_mod, R.cpu())

dev = R.cpu()

def t(arr: torch.Tensor):
    np_arr = arr.numpy()
    tvm_arr = R.empty(np_arr.shape, str(np_arr.dtype), dev)
    tvm_arr.copyfrom(np_arr)
    return tvm_arr

# The compiled function expects 2D input [SEQ, DIM] — the ellipsis in MDL
# represents a single leading batch dimension, not arbitrary leading dims.
x_2d = x_torch.squeeze(0)                             # [SEQ, DIM]

result = vm["TransformerEncoder__forward"](
    t(x_2d),             # [0]  x
    t(wQ),               # [1]  weightQuery.W
    t(bQ),               # [2]  weightQuery.b
    t(wK),               # [3]  weightKey.W
    t(bK),               # [4]  weightKey.b
    t(wV),               # [5]  weightValue.W
    t(bV),               # [6]  weightValue.b
    t(wO),               # [7]  weightOut.W
    t(bO),               # [8]  weightOut.b
    t(mha_norm_gamma),   # [9]  mhaNorm.gamma
    t(mha_norm_beta),    # [10] mhaNorm.beta
    t(wI),               # [11] weightIntermed.W
    t(bI),               # [12] weightIntermed.b
    t(wIO),              # [13] weightIntermedOut.W
    t(bIO),              # [14] weightIntermedOut.b
    t(ffn_norm_gamma),   # [15] ffnNorm.gamma
    t(ffn_norm_beta),    # [16] ffnNorm.beta
)

mdl_output = result.numpy()  # [SEQ, DIM]
print(f"MDL output shape: {mdl_output.shape}, mean={mdl_output.mean():.6f}")

# ---------------------------------------------------------------------------
# 4. Compare
# ---------------------------------------------------------------------------
hf_np = hf_output.numpy()
if hf_np.ndim == 3:
    hf_np = hf_np[0]   # drop batch dim if HF returns [B, T, D]
abs_diff = np.abs(mdl_output - hf_np)
rel_diff = abs_diff / (np.abs(hf_np) + 1e-8)

print(f"\nMax absolute difference : {abs_diff.max():.6f}")
print(f"Mean absolute difference: {abs_diff.mean():.6f}")
print(f"Max relative difference : {rel_diff.max():.4%}")

atol = 1e-4
if np.allclose(mdl_output, hf_np, atol=atol):
    print(f"\nPASS — outputs match within atol={atol}")
else:
    mismatch = np.sum(~np.isclose(mdl_output, hf_np, atol=atol))
    total = mdl_output.size
    print(f"\nFAIL — {mismatch}/{total} elements differ beyond atol={atol}")
    print("\nFirst differing positions (up to 5):")
    idxs = np.argwhere(~np.isclose(mdl_output, hf_np, atol=atol))
    for idx in idxs[:5]:
        print(f"  [{tuple(idx)}]  mdl={mdl_output[tuple(idx)]:.6f}  hf={hf_np[tuple(idx)]:.6f}")

# ---------------------------------------------------------------------------
# 5. Step-by-step NumPy reference to locate divergence
# ---------------------------------------------------------------------------
print("\n--- NumPy step-by-step reference ---")

x = x_2d.numpy().astype(np.float32)          # [T, 768]
wQ_n = wQ.numpy(); bQ_n = bQ.numpy()
wK_n = wK.numpy(); bK_n = bK.numpy()
wV_n = wV.numpy(); bV_n = bV.numpy()
wO_n = wO.numpy(); bO_n = bO.numpy()
wI_n = wI.numpy(); bI_n = bI.numpy()
wIO_n = wIO.numpy(); bIO_n = bIO.numpy()
gamma_mha = mha_norm_gamma.numpy(); beta_mha = mha_norm_beta.numpy()
gamma_ffn = ffn_norm_gamma.numpy(); beta_ffn = ffn_norm_beta.numpy()

H, D_k, D_v = 12, 64, 64

def linear_np(x, W, b):
    return x @ W.T + b

def layernorm_np(x, gamma, beta, eps=1e-12):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    sigma = np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + eps)
    return xc / sigma * gamma + beta

def gelu_np(x):
    return x * 0.5 * (1.0 + np.vectorize(
        lambda v: float(torch.erf(torch.tensor(v / np.sqrt(2)))))(x))

def softmax_np(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# Q, K, V projections
Q = linear_np(x, wQ_n, bQ_n)     # [T, 768]
K = linear_np(x, wK_n, bK_n)
V = linear_np(x, wV_n, bV_n)

# Compare projections with HF
with torch.no_grad():
    hf_Q = layer.attention.self.query(x_torch.squeeze(0)).numpy()
    hf_K = layer.attention.self.key(x_torch.squeeze(0)).numpy()
    hf_V = layer.attention.self.value(x_torch.squeeze(0)).numpy()
print(f"Q diff from HF: {np.abs(Q - hf_Q).max():.2e}")
print(f"K diff from HF: {np.abs(K - hf_K).max():.2e}")
print(f"V diff from HF: {np.abs(V - hf_V).max():.2e}")

# Reshape to multi-head: [T, H*D] → [H, T, D]
Q_h = Q.reshape(SEQ, H, D_k).transpose(1, 0, 2)   # [H, T, D_k]
K_h = K.reshape(SEQ, H, D_k).transpose(1, 0, 2)
V_h = V.reshape(SEQ, H, D_v).transpose(1, 0, 2)

# Attention scores + softmax
scores = np.einsum('hid,hjd->hij', Q_h, K_h) / np.sqrt(D_k)   # [H, T, T]
attn_w = softmax_np(scores)                                      # [H, T, T]

# Context
context = np.einsum('hij,hjd->hid', attn_w, V_h)  # [H, T, D_v]

# Merge heads: [H, T, D_v] → [T, H*D_v]
context_merged = context.transpose(1, 0, 2).reshape(SEQ, H * D_v)

# Output projection + residual + LN
attn_out = linear_np(context_merged, wO_n, bO_n)   # [T, 768]
attn_res  = attn_out + x
attn_norm = layernorm_np(attn_res, gamma_mha, beta_mha)

# Compare attn_norm with HF intermediate
with torch.no_grad():
    hf_attn_norm = layer.attention.output.LayerNorm(
        layer.attention.output.dense(
            layer.attention.self(x_torch.squeeze(0).unsqueeze(0),
                                  attention_mask=attn_mask_extended)[0]
        ) + x_torch.squeeze(0)
    ).numpy()
print(f"attn_norm diff from HF: {np.abs(attn_norm - hf_attn_norm).max():.2e}")

# FFN
ffn_intermed = linear_np(attn_norm, wI_n, bI_n)    # [T, 3072]
# Use torch for GELU to match HF exactly
with torch.no_grad():
    ffn_intermed_gelu = torch.nn.functional.gelu(torch.from_numpy(ffn_intermed)).numpy()
ffn_out = linear_np(ffn_intermed_gelu, wIO_n, bIO_n)   # [T, 768]
final = layernorm_np(ffn_out + attn_norm, gamma_ffn, beta_ffn)

print(f"final  diff from HF: {np.abs(final - hf_np).max():.2e}")
print(f"MDL vs NumPy ref:    {np.abs(mdl_output - final).max():.2e}")

# ---------------------------------------------------------------------------
# 6. Benchmark: PyTorch vs MDL-compiled
# ---------------------------------------------------------------------------
import time

WARMUP = 100
RUNS   = 300

# Weights as pre-converted TVM arrays (built once, reused every iteration)
_tvm_weights = [
    t(wQ), t(bQ), t(wK), t(bK), t(wV), t(bV),
    t(wO), t(bO),
    t(mha_norm_gamma), t(mha_norm_beta),
    t(wI), t(bI), t(wIO), t(bIO),
    t(ffn_norm_gamma), t(ffn_norm_beta),
]

def bench_pt(seq):
    x = torch.randn(1, seq, DIM)
    mask = torch.zeros(1, 1, 1, seq)
    with torch.no_grad():
        for _ in range(WARMUP):
            layer(x, attention_mask=mask)
    samples = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            layer(x, attention_mask=mask)
            samples.append((time.perf_counter() - t0) * 1e3)
    return np.array(samples)

def bench_mdl(seq):
    x_in = t(torch.randn(seq, DIM))
    args = [x_in] + _tvm_weights
    fn   = vm["TransformerEncoder__forward"]
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1e3)
    return np.array(samples)

def fmt(s):
    mn, med, p95 = np.min(s), np.median(s), np.percentile(s, 95)
    return f"min={mn:.3f}  med={med:.3f}  p95={p95:.3f}"

print(f"\n--- Benchmark ({WARMUP} warmup, {RUNS} timed runs, times in ms) ---")

for seq in [8, 32, 128, 512]:
    print(f"\n  SEQ={seq}")
    pt_s = bench_pt(seq)
    print(f"    PyTorch : {fmt(pt_s)}", flush=True)
    try:
        mdl_s = bench_mdl(seq)
        ratio = np.median(pt_s) / np.median(mdl_s)
        tag   = "MDL faster" if ratio >= 1 else "PyTorch faster"
        print(f"    MDL/TVM : {fmt(mdl_s)}  ({ratio:.2f}x {tag})", flush=True)
    except Exception as e:
        print(f"    MDL/TVM : skipped ({e})", flush=True)

# -- isolate data-conversion overhead --
print("\n--- Data-conversion overhead (torch→TVM, not included above) ---")
_x_np = torch.randn(8, DIM).numpy()
_conv_samples = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    arr = R.empty(_x_np.shape, 'float32', dev)
    arr.copyfrom(_x_np)
    _conv_samples.append((time.perf_counter() - t0) * 1e3)
print(f"  1 tensor [8,768] copy: med={np.median(_conv_samples):.3f} ms")

# ---------------------------------------------------------------------------
# 7. Per-kernel profiling at SEQ=512
# ---------------------------------------------------------------------------
print("\n--- Per-kernel profiling (SEQ=512, 50 runs each) ---")

PROF_SEQ = 512
x_prof = t(torch.randn(PROF_SEQ, DIM))
args_prof = [x_prof] + _tvm_weights

# Warm up
fn = vm["TransformerEncoder__forward"]
for _ in range(20):
    fn(*args_prof)

# Instrument: record wall-clock time per kernel call
import collections
kernel_times = collections.defaultdict(list)
_t_before = {}

def _instrument(func, func_symbol, before_run, ret_value, *args):
    if before_run:
        _t_before[func_symbol] = time.perf_counter()
    else:
        elapsed = (time.perf_counter() - _t_before.get(func_symbol, time.perf_counter())) * 1e6
        kernel_times[func_symbol].append(elapsed)
    return None  # continue normally

vm.set_instrument(_instrument)

PROF_RUNS = 50
for _ in range(PROF_RUNS):
    fn(*args_prof)

vm.set_instrument(None)  # remove instrument

# Aggregate
print(f"\n  {'Kernel':<55} {'calls':>5}  {'med µs':>8}  {'sum ms':>8}")
print(f"  {'-'*55}  {'-'*5}  {'-'*8}  {'-'*8}")
total_accounted = 0.0
rows = []
for sym, times_list in kernel_times.items():
    per_call = np.array(times_list).reshape(PROF_RUNS, -1).mean(axis=0)  # avg over runs
    med = float(np.median(per_call))
    calls_per_run = len(per_call)
    total = med * calls_per_run / 1e3  # ms
    total_accounted += total
    rows.append((sym, calls_per_run, med, total))

rows.sort(key=lambda r: r[3], reverse=True)
for sym, calls, med, total in rows[:30]:
    name = sym if len(sym) <= 55 else sym[:52] + "..."
    print(f"  {name:<55} {calls:>5}  {med:>8.1f}  {total:>8.3f}")
print(f"  {'TOTAL accounted':<55} {'':>5}  {'':>8}  {total_accounted:>8.3f}")

# Also time with time_evaluator for ground truth
te = vm.time_evaluator("TransformerEncoder__forward", dev, number=50, repeat=3)
te_result = te(*args_prof)
print(f"\n  time_evaluator (50 runs×3 repeats): mean={te_result.mean*1e3:.3f} ms  "
      f"std={te_result.std*1e3:.3f} ms")