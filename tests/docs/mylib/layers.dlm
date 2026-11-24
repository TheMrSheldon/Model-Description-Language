/**
 * @file
 * @brief Layers of the test package (see tests/CMakeLists.txt, mdl_doc_golden).
 */
/**
 * @brief A scaled identity, as in \cite layernorm.
 * @param dim the size of the last axis
 */
public module Scale[dim: uint] {
    /** @brief The factors. */
    public w: float[dim]
    hidden: float[dim]

    /** @brief Scales @p x. @return \f$x \odot w\f$ */
    public float[..., dim] forward(x: float[..., dim]) => x * w;
}

/** @brief Doubles @p x. */
public float[...] twice(x: float[...]) => scale(x, 2.0);
/** @brief Doubles a masked @p x (an overload: its own anchor). */
public masked float[...] twice(x: masked float[...]) => scale(x, 2.0);

float[...] internal(x: float[...]) => x;

/** Re-exported from the implementation. */
public from ._impl import scale;
