"""
Benchmarks TransformerEncoder__forward compiled to IREE (.vmfb) vs PyTorch.

Mirrors validate_bert.py but uses the IREE Python runtime instead of TVM.

Usage:
    python benchmark_iree.py [path/to/transformer_iree.vmfb]

The .vmfb defaults to /tmp/transformer_iree.vmfb.  Compile it with:
    mdlc examples/02_transformer/model.dlm \\
        -a examples/02_transformer/model.yml --backend iree -o /tmp/transformer_iree
"""

import sys
import os
import time
import numpy as np
import torch
from transformers import BertModel, BertConfig
import iree.runtime as rt

VMFB_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/transformer_iree.vmfb"

# ---------------------------------------------------------------------------
# 1. Load HuggingFace BERT layer-0 weights
# ---------------------------------------------------------------------------
print("Loading bert-base-uncased ...")
config = BertConfig()
model = BertModel(config)
model.eval()

layer = model.encoder.layer[0]

wQ = layer.attention.self.query.weight.detach().float()
bQ = layer.attention.self.query.bias.detach().float()
wK = layer.attention.self.key.weight.detach().float()
bK = layer.attention.self.key.bias.detach().float()
wV = layer.attention.self.value.weight.detach().float()
bV = layer.attention.self.value.bias.detach().float()
wO = layer.attention.output.dense.weight.detach().float()
bO = layer.attention.output.dense.bias.detach().float()
mha_norm_gamma = layer.attention.output.LayerNorm.weight.detach().float()
mha_norm_beta  = layer.attention.output.LayerNorm.bias.detach().float()
wI  = layer.intermediate.dense.weight.detach().float()
bI  = layer.intermediate.dense.bias.detach().float()
wIO = layer.output.dense.weight.detach().float()
bIO = layer.output.dense.bias.detach().float()
ffn_norm_gamma = layer.output.LayerNorm.weight.detach().float()
ffn_norm_beta  = layer.output.LayerNorm.bias.detach().float()

# ---------------------------------------------------------------------------
# 2. Load IREE module
# ---------------------------------------------------------------------------
if not os.path.exists(VMFB_PATH):
    print(f"VMFB not found: {VMFB_PATH}")
    sys.exit(1)

print(f"Loading {VMFB_PATH} ...")
iree_config = rt.Config("local-task")
iree_ctx = rt.SystemContext(config=iree_config)
with open(VMFB_PATH, "rb") as f:
    vmfb_data = f.read()
iree_ctx.add_vm_module(rt.VmModule.copy_buffer(iree_ctx.instance, vmfb_data))
iree_fn = iree_ctx.modules.module["TransformerEncoder__TransformerEncoder__forward"]
iree_dev = iree_config.device

def to_iree(t: torch.Tensor):
    return rt.asdevicearray(iree_dev, t.numpy())

# Pre-convert weights once
_iree_weights = [
    to_iree(wQ), to_iree(bQ),
    to_iree(wK), to_iree(bK),
    to_iree(wV), to_iree(bV),
    to_iree(wO), to_iree(bO),
    to_iree(mha_norm_gamma), to_iree(mha_norm_beta),
    to_iree(wI),  to_iree(bI),
    to_iree(wIO), to_iree(bIO),
    to_iree(ffn_norm_gamma), to_iree(ffn_norm_beta),
]

# ---------------------------------------------------------------------------
# 3. Correctness check at SEQ=8
# ---------------------------------------------------------------------------
BATCH, SEQ, DIM = 1, 8, 768
torch.manual_seed(42)
x_torch = torch.randn(BATCH, SEQ, DIM)
attn_mask = torch.zeros(BATCH, 1, 1, SEQ)

with torch.no_grad():
    hf_out = layer(x_torch, attention_mask=attn_mask).squeeze(0).numpy()

x_2d = x_torch.squeeze(0)
iree_result = iree_fn(to_iree(x_2d), *_iree_weights)
mdl_out = np.array(iree_result, copy=True)

abs_diff = np.abs(mdl_out - hf_out)
print(f"\nCorrectness check (SEQ=8):")
print(f"  Max absolute diff : {abs_diff.max():.6f}")
print(f"  Mean absolute diff: {abs_diff.mean():.6f}")
atol = 1e-3
if np.allclose(mdl_out, hf_out, atol=atol):
    print(f"  PASS (atol={atol})")
else:
    n_bad = np.sum(~np.isclose(mdl_out, hf_out, atol=atol))
    print(f"  FAIL — {n_bad}/{mdl_out.size} elements differ beyond atol={atol}")

# ---------------------------------------------------------------------------
# 4. Benchmark
# ---------------------------------------------------------------------------
WARMUP = 50
RUNS   = 200

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

def bench_iree(seq):
    x_in = to_iree(torch.randn(seq, DIM))
    args = [x_in] + _iree_weights
    for _ in range(WARMUP):
        iree_fn(*args)
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        iree_fn(*args)
        samples.append((time.perf_counter() - t0) * 1e3)
    return np.array(samples)

def fmt(s):
    return f"min={np.min(s):.3f}  med={np.median(s):.3f}  p95={np.percentile(s,95):.3f}"

print(f"\n--- Benchmark ({WARMUP} warmup, {RUNS} timed runs, ms) ---")

for seq in [8, 32, 128, 512, 1024]:
    print(f"\n  SEQ={seq}")
    pt_s = bench_pt(seq)
    print(f"    PyTorch  : {fmt(pt_s)}")
    iree_s = bench_iree(seq)
    ratio = np.median(pt_s) / np.median(iree_s)
    tag = "IREE faster" if ratio >= 1 else "PyTorch faster"
    print(f"    IREE/MDL : {fmt(iree_s)}  ({ratio:.2f}x {tag})")
