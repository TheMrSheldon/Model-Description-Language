#!/usr/bin/env python3
"""The PyTorch reference of the benchmark protocol (scripts/benchmark.sh): the HuggingFace BERT encoder (BERT-base: 12
layers, hidden size 768, 12 heads, FFN size 3072; tiny: model_grad.yml), inference and training step (forward and
backward with respect to the weights and the input), float32 and bfloat16, batch 8 x sequence 128 (tiny: 8 x 32), 3
warm-up and 20 timed runs; prints the medians in the format of bench.cpp.

Usage: scripts/benchmark_torch.py [--device cuda|cpu] [--compile] [--attention eager|sdpa] [--sizes base tiny]
  --attention: transformers' attention implementation: eager (matmul, softmax, matmul; the default, as MDL's model) or
  sdpa (torch's scaled_dot_product_attention: fused attention kernels).
Needs torch and transformers.
"""

import argparse
import statistics
import time

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

SIZES = {
    "base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, seq=128),
    "tiny": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, seq=32),
}


def bench(name, step, device, warmup, iters):
    for _ in range(warmup):
        step()
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    print(f"{name} median {times[len(times) // 2]:.4f} min {times[0]:.4f} ({iters})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="torch.compile the encoder")
    parser.add_argument("--attention", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--sizes", nargs="+", default=["base", "tiny"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=None, help="sequence length (default: 128, tiny: 32)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    for size in args.sizes:
        cfg = dict(SIZES[size])
        seq = cfg.pop("seq")
        if args.seq is not None:
            seq = args.seq
        # Dropout off, as in the MDL model.
        config = BertConfig(**cfg, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        config._attn_implementation = args.attention
        for mode in ("infer", "train"):
            for precision, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
                torch.manual_seed(7)
                encoder = BertEncoder(config).to(args.device, dtype)
                encoder.train(mode == "train")
                run = torch.compile(encoder) if args.compile else encoder
                x = torch.randn(args.batch, seq, cfg["hidden_size"], device=args.device, dtype=dtype) * 0.02
                if mode == "infer":
                    def step():
                        with torch.no_grad():
                            run(x)
                else:
                    x.requires_grad_(True)
                    dy = torch.randn_like(x) * 0.02

                    def step():
                        y = run(x).last_hidden_state
                        y.backward(dy)

                suffix = ("_compile" if args.compile else "") + ("_sdpa" if args.attention == "sdpa" else "")
                bench(f"torch_{size}_{mode}_{precision}_{args.device}{suffix}", step, args.device, args.warmup,
                      args.iters)


if __name__ == "__main__":
    main()
