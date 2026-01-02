# pip install apache-tvm-ffi

import tvm_ffi
mod = tvm_ffi.load_module("/workspaces/model-description-language/build/examples/03_pytorch/lib03_pytorch.so")

import torch
W = torch.arange(3*8, dtype=torch.float32).reshape(3,8)
b = torch.arange(3, dtype=torch.float32)

x = torch.arange(8, dtype=torch.float32)
print("Expected output:", W@x + b)

outbuf = torch.empty(3, dtype=torch.float32)
mod.forward(b, W, x, outbuf)
print("Output:         ", outbuf)
