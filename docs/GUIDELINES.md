## IRON-FI Experiments and Cursor Helper Guidelines

This document captures the provided implementation guidance:

- Build paper-consistent experiments with the resolvent/prox step at the core.
- Prefer Cholesky/solvers over explicit matrix inverses.
- Update `gamma_k` each iteration via `gamma_{k+1}=(gamma_k+\alpha\mu)/(1+\alpha)`.
- Inject noise as a perturbation of the prox center: `xi = (sqrt(alpha)/(1+tau)) * sigma * eta`.
- Keep memory usage low: store only thin slices for plotting.

Suggested layout:

```
ironfi/
  resolvent.py
  gamma.py
  noise.py
  __init__.py
experiments/
  quad_iron_fi.py
  nonconvex_iron_fi_jax.py
plots/
  utils.py
docs/
  ironfi_paper.tex
  GUIDELINES.md
```

Plot conventions: log-scale for mean error and 2D projections of initial/final clouds. Use deterministic seeds and CSV logging for reproducibility.


