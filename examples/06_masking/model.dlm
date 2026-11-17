import std;

/** Average over the last dimension (generic: a masked argument averages the valid elements only). */
float[...] average(x: float[...]) => mean(x);

/**
 * Average over the last dimension of a masked tensor, written explicitly with its mask: `x.data` has zeros at the
 * masked elements and `x.mask` is 1 at the valid ones, so this is the sum of the valid elements over their number.
 */
float[...] average(x: masked float[...]) => mean(x.data) / mean(x.mask);

/**
 * @brief Averages the vectors of a masked sequence (`...` is the sequence; its masked elements are ignored).
 * `withOverload` uses the masked overload of `average`, `generic` the generic masking of `mean`; both must agree.
 * `meanAll` and `sumAll` reduce all valid elements to a scalar.
 */
module Pool[dim: uint] {
    float[dim] withOverload(x: masked float[..., dim]) => average(x[... d -> d ...]);
    float[dim] generic(x: masked float[..., dim]) => mean(x[... d -> d ...]);
    /** The mean and sum of all valid elements (the flattened tensor: its masked rows are ignored). */
    float meanAll(x: masked float[..., dim]) => mean_flat(x);
    float sumAll(x: masked float[..., dim]) => sum_flat(x);
}
