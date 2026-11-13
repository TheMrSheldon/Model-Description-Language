module Scale[n: uint] {
    s: float[n]
    float[..., n] forward(x: float[..., n]) => x * s;
}
