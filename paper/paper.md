---
title: 'FDNkit: a fractional dynamical-network and multifractal toolkit for intracranial EEG'
tags:
  - Python
  - neuroscience
  - intracranial EEG
  - iEEG
  - detrended fluctuation analysis
  - multifractal
  - fractional-order dynamics
  - network physiology
authors:
  - name: Samir Hossain
    orcid: 0009-0003-8986-0946
    affiliation: 1
affiliations:
  - name: Department of Electrical and Computer Engineering, Texas Tech University, Lubbock, TX, USA
    index: 1
date: 5 September 2026
bibliography: paper.bib
---

# Summary

`FDNkit` is an open-source Python library and command-line tool that turns
intracranial-EEG (iEEG) recordings into **fractal** and
**fractional-dynamical-network** features and evaluates them with statistically
honest, subject-aware cross-validation. It implements three complementary
analyses on multi-channel time series: monofractal detrended fluctuation
analysis (DFA) for the Hurst exponent [@Peng1994]; multifractal DFA (MFDFA) for
the generalized Hurst exponent `h(q)`, the multifractal width, and the
singularity spectrum `f(α)` [@Kantelhardt2002]; and a fractional-order dynamical
network (FODN) model that estimates each channel's fractional order, a sparse
directed coupling matrix, and eigenvector-based hub scores [@Gupta2018;
@Xue2017]. Features are assembled into tidy, one-row-per-trial `pandas`
DataFrames, and a built-in logistic-regression harness reports leave-one-subject-out
performance with a subject-level permutation test and bootstrap confidence
intervals. Plotting utilities, EDF/HDF5 readers, and a synthetic iEEG generator
round out the package, and a test suite validates the numerical core against
signals with known scaling properties.

# Statement of need

Fractal and fractional-dynamics descriptors of brain activity (long-range
temporal correlations, multifractality, and fractional-order network coupling)
are increasingly used to characterize cognition and pathology in iEEG
[@Hardstone2012; @Gupta2018]. Yet the analysis code behind such studies is
typically bespoke: tied to one dataset, entangled with a graphical interface,
and not reusable.

Several mature Python packages already compute the *monofractal and multifractal*
half of this picture. `nolds` [@Scholzel2019] provides DFA alongside other
nonlinear measures; `MFDFA` [@RydinGorjao2022] and `fathon` [@Bianchi2020]
implement multifractal DFA and its variants efficiently (building on the widely
used MATLAB tutorial implementation of @Ihlen2012); and `neurokit2`
[@Makowski2021] wraps DFA/MFDFA within a broad biosignal-processing toolkit.
FDNkit does **not** aim to displace these for plain DFA/MFDFA: its own DFA/MFDFA
core is validated against them and against ground-truth signals. What is missing
from all of them is the **fractional-order dynamical-network (FODN)** side:
estimating a per-channel fractional order, a sparse directed coupling matrix, and
eigenvector-based hub structure from multi-channel recordings [@Gupta2018;
@Xue2017]. Likewise, general EEG suites such as EEGLAB [@Delorme2004],
MNE-Python [@Gramfort2013], and FieldTrip [@Oostenveld2011] excel at IO,
preprocessing, and classical connectivity but implement neither multifractal DFA
nor fractional-order network estimation.

`FDNkit` fills that gap by combining three things no existing package offers
together: (i) multifractal DFA, (ii) fractional-order dynamical-network coupling,
and (iii) an iEEG-oriented pipeline that windows labeled trials, assembles tidy
feature tables, and classifies them. It generalizes a single-purpose research
tool, built by the author to compute the features validated in [@Beeram2026]
(which the author co-authored), into a reusable library with a stable API,
documentation, and tests, so that other groups can compute the same features on
their own iEEG.

Just as important, it addresses a methodological pitfall that motivated its
creation. In the original study, trials from the same patient appeared in both
training and test folds; because patient identity is highly predictive, this
row-wise cross-validation inflated the reported accuracy. `FDNkit` therefore
**defaults to leave-one-subject-out cross-validation and requires a group
label**, offers a group-aware permutation test, and marks trial-wise evaluation
explicitly as *optimistic*. Making honest evaluation the path of least
resistance is a deliberate design choice, and one that the general-purpose
fractal packages above, which return exponents but leave evaluation entirely to
the user, do not provide.

# Functionality

The package is organized around array-first, pure-function cores with pandas,
IO, and plotting layered on top:

- **`fdnkit.dfa`**: `dfa(signal)` returns the Hurst exponent from the log–log
  slope of the detrended fluctuation function.
- **`fdnkit.mfdfa`**: `mfdfa(signal)` returns `h(q)`, the multifractal width
  `Δh`, and, via a Legendre transform, the singularity spectrum `f(α)`.
- **`fdnkit.fodn`**: `fit_fodn(signals)` estimates per-channel fractional orders
  from a Haar-wavelet variance regression, builds the Grünwald–Letnikov
  fractional-derivative signal, and fits a sparse coupling matrix by regularized
  least squares refined with an ADMM-LASSO unknown-input step; it exposes the
  leading eigenvalue (spectral radius), the dominant eigenvector (hub scores),
  and network sparseness.
- **`fdnkit.features`**: `extract_features` and `feature_table` assemble tidy
  DataFrames, including a five-feature "core" set that reproduces the reference
  study.
- **`fdnkit.classify`**: `classify` / `classify_dataframe` run subject-wise
  cross-validation with permutation testing and bootstrap intervals.
- **`fdnkit.io`, `fdnkit.preprocessing`, `fdnkit.viz`, `fdnkit.synthetic`**:
  EDF (via MNE-Python) and HDF5 readers, z-scoring / bad-channel flagging /
  windowing, plots (fluctuation curves, `h(q)`, spectra, coupling heatmaps, hub
  bars), and generators for fractional Gaussian noise, multiplicative cascades,
  and coupled multi-channel signals.

A command-line interface (`fdnkit extract`, `fdnkit classify`, `fdnkit demo`)
covers the common batch workflow, and `fdnkit demo` reproduces a full
extract-and-classify pipeline on synthetic data with no downloads.

# Validation

`FDNkit`'s numerical core is verified against ground truth rather than only
against itself. Using exact Davies–Harte synthesis of fractional Gaussian noise
[@Davies1987], DFA recovers the target Hurst exponent across `H = 0.3–0.9` to
within a few hundredths; white noise yields `H ≈ 0.5` and Brownian motion
`H ≈ 1.5`. MFDFA returns a wide `h(q)` for a multiplicative binomial cascade and
a narrow one for a monofractal signal, and `h(q=2)` matches the DFA Hurst
exponent to numerical precision. The FODN estimator recovers finite fractional
orders, coupling matrices, and hub scores on synthetic coupled systems. Negative
`q` moments in MFDFA are known to be destabilized by segments of near-zero
detrended variance, a documented source of spurious multifractality
[@Ludescher2011]; `FDNkit` guards against this with an optional scale-relative
fluctuation floor that leaves well-behaved signals unchanged. These
checks run in continuous integration across Python 3.9–3.12. A worked example
(`examples/real_data_eegbci.py`) runs the full pipeline on a public EDF recording
from the PhysioNet EEG Motor Movement/Imagery dataset [@Schalk2004; @Goldberger2000],
demonstrating the toolkit on genuine data.

# References
