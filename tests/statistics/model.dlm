import std;

/** @brief The variance of a scaled input (tests/statistics): population (dof 0) and sample (dof 1) variance. */
module Stats[] {
    w: float[8]

    public float[...] population(x: float[..., 8]) => var(x * w, 0);
    public float[...] sample(x: float[..., 8]) => var(x * w, 1);
}
