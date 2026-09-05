# Contributing to FDNkit

Thanks for your interest in improving FDNkit! Bug reports, feature requests, and
pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/SamirHossain099/fdnkit
cd fdnkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check src tests        # lint
pytest --cov=fdnkit         # tests + coverage
```

Please:

- Keep the numerical core **array-first and pure**: pandas/IO/plotting layer on
  top, and matplotlib/MNE/h5py stay optional (lazily imported).
- Add a test for any new behavior. Where possible, validate against **ground
  truth** (a signal with a known Hurst exponent, a cascade with a known
  multifractal width), not just against the current output.
- Match the existing docstring style (NumPy-format) and keep public functions
  documented.
- Do **not** commit data. The private clinical cohort must never be added;
  examples and tests use `fdnkit.synthetic`.

## Reporting bugs

Open an issue with a minimal reproducer, ideally one built from
`fdnkit.synthetic` so it runs anywhere.

## Scope

FDNkit is intentionally focused on fractal / fractional-dynamical-network methods
for iEEG. General EEG preprocessing, montages, and filtering belong in
MNE-Python; connectivity beyond FODN (Granger, transfer entropy) and a result
database are candidate future additions but out of scope for the core.
