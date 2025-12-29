# pip install apache-tvm-ffi

import tvm_ffi
mod = tvm_ffi.load_module("/workspaces/model-description-language/build/examples/01_ffn/libtest.so")

import torch
x = torch.ones((128, 128), dtype=torch.float32, device="cpu")
print(x.shape)
y = torch.empty_like(x)
z = torch.empty_like(x)
w = torch.empty_like(x)
print(y)
mod.forward(x, y, z, w)
print(y)
