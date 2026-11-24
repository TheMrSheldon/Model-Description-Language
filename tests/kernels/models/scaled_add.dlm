// A model for the kernel-package tests (tests/kernels/matching).
module M[] {
    public float[..., 8] forward(x: float[..., 8], y: float[..., 8]) => x * 2.0 + y;
    public float[..., 8] other(x: float[..., 8], y: float[..., 8]) => x * 3.0 + y;
    public float[..., 8] inner(x: float[..., 8], y: float[..., 8]) => (x * 3.0 + y) * 0.5;
    public float[..., 8] escaped(x: float[..., 8], y: float[..., 8]) {
        auto t = x * 2.0;
        return t + y + t;
    }
}
