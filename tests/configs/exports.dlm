// The compiled functions (tests/CMakeLists.txt): the public member functions, or those that --export names.
module Ex[] {
    W: float[4, 4]
    float[..., 4] helper(x: float[..., 4]) => x @ W.T();
    public float[..., 4] forward(x: float[..., 4]) => helper(x);
    public float loss(x: float[..., 4]) => mean_flat(forward(x));
}
