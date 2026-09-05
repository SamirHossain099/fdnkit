# Methods

## Detrended fluctuation analysis (DFA)

For a signal `x`, DFA computes the integrated profile `Y = cumsum(x - mean(x))`,
partitions it into non-overlapping windows of size `s`, removes a low-order
polynomial trend in each window, and measures the root-mean-square residual
`F(s)`. The Hurst exponent `H` is the slope of `log2 F(s)` against `log2 s`.
Values `H ≈ 0.5` indicate white noise, `H > 0.5` persistent (long-range)
correlations, and `H > 1` non-stationary (motion-like) behavior.

## Multifractal DFA (MFDFA)

MFDFA generalizes the second step to arbitrary moment orders `q`:

$$ F_q(s) = \left(\frac{1}{N_s}\sum_v \mathrm{RMS}_v(s)^{\,q}\right)^{1/q}, \qquad
   F_0(s) = \exp\!\left(\tfrac12 \langle \ln \mathrm{RMS}_v(s)^2\rangle\right). $$

The generalized Hurst exponent `h(q)` is the log–log slope of `F_q(s)` vs `s`.
A signal is **monofractal** when `h(q)` is constant and **multifractal** when it
decreases with `q`; the width `Δh = max h(q) − min h(q)` quantifies this. A
Legendre transform yields the singularity spectrum `f(α)` (`fdnkit.mfdfa.multifractal_spectrum`).

FDNkit uses a forward (non-overlapping) partition and the default grids
`scales = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]` and
`q = [-5, -3, -2, -1, 0, 1, 2, 3, 5]`, matching the validated reference
implementation.

## Fractional-order dynamical network (FODN)

FODN models the multi-channel signal as a sparsely-coupled fractional-order
linear system. For each channel it:

1. estimates a **fractional order** `α` from the slope of Haar-wavelet detail
   variance across dyadic scales;
2. forms the **Grünwald–Letnikov fractional difference** `z` using the truncated
   kernel `Γ(-α + j) / [Γ(-α) Γ(j+1)]`;
3. fits a **coupling matrix** `A` from `z_k ≈ A x_{k-1}` by regularized least
   squares, then refines it with an ADMM-LASSO unknown-input step that promotes
   sparsity.

From `A`, FDNkit reports the **leading eigenvalue** (spectral radius, a
network-gain / stability proxy), the **dominant eigenvector** magnitude
(per-channel hub scores), and **sparseness** (fraction of appreciable edges).

## Honest evaluation

Because trials from one patient are highly self-similar, evaluating a classifier
with row-wise splits lets patient identity leak from train to test and inflates
accuracy. FDNkit defaults to **leave-one-subject-out** cross-validation (whole
subjects held out), requires an explicit `groups` argument for it, and provides a
**group-aware permutation test** and **bootstrap confidence intervals**.
Trial-wise leave-one-out is available but flagged optimistic.

## Validation

The numerical core is tested against ground truth:

- DFA recovers the Hurst exponent of exact fractional Gaussian noise
  (Davies–Harte synthesis) across `H = 0.3–0.9`; white noise → `0.5`, Brownian
  motion → `1.5`.
- MFDFA reports a wide `h(q)` for a multiplicative binomial cascade and a narrow
  one for a monofractal signal; `h(q=2)` equals the DFA Hurst exponent exactly.
- FODN returns finite orders, coupling, and hubs on synthetic coupled systems.

## References

- Peng et al. (1994), *Phys. Rev. E* 49:1685.
- Kantelhardt et al. (2002), *Physica A* 316:87.
- Davies & Harte (1987), *Biometrika* 74:95.
- Gupta, Pequito & Bogdan (2018), *ACC*.
- Beeram et al. (2026), *Front. Netw. Physiol.* 6:1768476.
