# Builds a shared library plus PyTorch bindings (model_pytorch.py) that main.py loads via tvm_ffi.
mdl_add_model(03_pytorch SOURCE model.dlm CONFIG model.yaml SHARED BINDINGS python-torch)
