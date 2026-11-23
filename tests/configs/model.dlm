module Scale[n: uint] {
    s: float[n]
    public float[..., n] forward(x: float[..., n]) => x * s;
}

module Project[n: uint] {
    W: float[n, n]
    s: float[n]
    public float[..., n] forward(x: float[..., n]) => (x @ W.T()) * s;
}
