# IRON-FI: Implicit Resolvent Optimization under Noise

This repository contains a paper-consistent implementation of **IRON-FI** (fully implicit resolvent / Backward–Euler discretization) and the experimental suite used in the IRON preprint.

Core IRON-FI definitions (matching the paper):
- **Center**: $c_k = (v_k + \tau_k x_k)/(1+\tau_k)$
- **Parameters**: $\tau_k = 1/\alpha_k + \mu/\gamma_k$, $\lambda_k = \alpha_k/(\gamma_k(1+\tau_k))$
- **Noise as center perturbation**: $\xi_k = (\sqrt{\alpha_k}/(1+\tau_k))\ \sigma\ \eta_k$, $\eta_k\sim\mathcal N(0,I)$
- **Resolvent step**: $x_{k+1} = \mathrm{prox}_{\lambda_k f}(c_k+\xi_k)$
- **State updates**: $v_{k+1}=x_{k+1} + (x_{k+1}-x_k)/\alpha_k$, $\gamma_{k+1}=(\gamma_k+\alpha_k\mu)/(1+\alpha_k)$

---

## Quickstart (reproduce figures)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Recommended: install the repo in editable mode (avoids PYTHONPATH)
pip install -e .

# Headless plotting + local Matplotlib cache
export MPLBACKEND=Agg
export MPLCONFIGDIR=$(pwd)/.mplcache
```

Then run any experiment command in the sections below.

---

## Project layout

```
datasets/
  mnist.py                     # download/cache MNIST into data/mnist/ and load NumPy arrays
docs/
  ironfi_paper.tex             # IRON paper draft
  nag_gs_paper.tex             # NAG-GS paper draft (pseudo-code)
experiments/
  quad_iron_fi.py              # quadratic IRON-FI (NumPy/SciPy)
  nonconvex_iron_fi_numpy.py   # nonconvex log-cosh IRON-FI (NumPy)
  nonconvex_iron_fi_jax.py     # nonconvex log-cosh IRON-FI (JAX prototype)
  logreg_synth_ironfi.py       # Part A: synthetic ridge-logistic regression (IRON-only)
  mnist_softmax_benchmark.py   # Part B: single-run MNIST benchmark + optional alpha tuning
  mnist_softmax_journal.py     # Part B: multi-seed journal runner (mean±std + averaged curves)
  mnist_softmax_core.py        # shared training loops for Part B runners
ironfi/
  resolvent.py                 # quadratic resolvent step helper
  ironfi.py                    # generic IRON-FI outer step (vector case, explicit center noise)
  inner_solvers.py             # LM/Newton inner solver (dense small-d) + stats
  ironfi_mf.py                 # matrix-free IRON-FI (Newton-CG) for MNIST softmax
  optimizers/
    nag_gs.py                  # NAG-GS update (paper pseudo-code)
    adamw.py                   # AdamW baseline (NumPy)
  noise.py                     # noise samplers
  gamma.py                     # gamma update helper
models/
  softmax.py                   # softmax regression loss/grad + Hessian-vector product
plots/
  utils.py                     # shared plotting utilities
  slices.py                    # 2D objective slices + projected clouds (nonconvex)
utils/
  expio.py                     # config + csv logging helpers
  plotting.py                  # small plotting helpers
requirements.txt
requirements-jax.txt           # optional (JAX prototype only)
pyproject.toml                 # packaging (pip install -e .)
```

Generated outputs:
- `figs/`: figures (PDFs)
- `logs/`: run logs (CSV/JSON)
- `data/`: datasets (MNIST cached under `data/mnist/`)

---

## Quadratic experiment (paper figure regeneration)

This generates two PDFs per \(\alpha\):
- `figs/quad_mean_alpha{α}.pdf`
- `figs/quad_clouds_alpha{α}.pdf`

```bash
python experiments/quad_iron_fi.py \
  --alpha-scale 1 10 20 500 \
  --nsamples 20000 --iters 100 --sigma 1.0 --seed 0 \
  --eigs 1.0 1.0 3.0 \
  --save-figs --no-show
```

---

## Nonconvex experiment (NumPy, log-cosh)

This generates (per \(\alpha\)):
- `figs/ncx_numpy_alpha{α}_mean_norm.pdf`
- `figs/ncx_numpy_alpha{α}_cloud.pdf`
- `figs/logcosh_slices_alpha{α}.pdf`

```bash
python experiments/nonconvex_iron_fi_numpy.py \
  --alpha-scale 1 10 200 500 \
  --nsamples 20000 --iters 10 --sigma 1.0 --seed 0 \
  --newton-it 6 --tol 1e-8 \
  --step-cap 0.5 --max-ls 12 --clip-x 30 \
  --plot-lim 5 \
  --save-figs --no-show
```

---

## Part A — Synthetic ridge-logistic regression (IRON-only)

This suite validates:
- stationary MSE scaling \(\widehat{\mathrm{MSE}}(\alpha)\sim 1/\alpha\) (slope close to \(-1\) on log–log),
- tolerance sweep showing \(\varepsilon\) does not need to shrink with \(\alpha\),
- mean inner LM/Newton iterations vs \(\alpha\).

```bash
python experiments/logreg_synth_ironfi.py \
  --n 20000 --d 50 --iters 1000 --burn-frac 0.3 \
  --alpha-grid 1 2 5 10 20 50 100 200 \
  --tol-grid 1e-2 1e-4 1e-6 \
  --sigma 1.0 --reg 1e-2 --seed 0 \
  --slope-fit-min-alpha 5 \
  --no-show
```

Outputs:
- `figs/synth_logreg_mse_vs_alpha_tol<best>.pdf`
- `figs/synth_logreg_tol_effect.pdf`
- `figs/synth_logreg_inner_iters_vs_alpha.pdf`

---

## Part B — MNIST softmax regression benchmark (IRON-FI vs NAG-GS vs AdamW)

MNIST is downloaded and cached under `data/mnist/` by the loader.

### MNIST download (first run)

The MNIST benchmark script can download MNIST automatically and cache it locally.

- **Where it is stored**: `data/mnist/`  
  - raw `.gz` files: `data/mnist/raw/`
  - cached arrays: `data/mnist/mnist.npz`

- **How to download (first run)**:

```bash
python experiments/mnist_softmax_benchmark.py --data-dir data/mnist --download --epochs 1 --no-show
```

After this first download, you can run all MNIST experiments **without** `--download` (they will reuse `data/mnist/mnist.npz`).

### Single-run benchmark (quick check)

```bash
python experiments/mnist_softmax_benchmark.py \
  --data-dir data/mnist --download \
  --epochs 10 --batch-size 256 --reg 1e-4 --seed 0 \
  --naggs-alpha 0.5 --naggs-mu 1 --naggs-gamma0 1 \
  --adamw-lr 1e-3 --adamw-weight-decay 0.0 \
  --ironfi-alpha 1 --ironfi-mu 1 --ironfi-gamma0 1 \
  --ironfi-inner-tol 1e-3 --ironfi-inner-newton 8 \
  --ironfi-cg-tol 1e-3 --ironfi-cg-max-it 200 \
  --no-show
```

### Grid search (10 epochs) for IRON-FI \(\alpha\)

```bash
python experiments/mnist_softmax_benchmark.py \
  --data-dir data/mnist \
  --epochs 10 --batch-size 128 --reg 1e-4 --seed 0 \
  --tune-ironfi --tune-epochs 10 \
  --ironfi-alpha-grid 0.75 1 1.25 1.5 2 2.5 3 \
  --ironfi-mu 1 --ironfi-gamma0 1 \
  --no-show
```

### Journal run (25 epochs, multi-seed, averaged curves + summary table)

```bash
python experiments/mnist_softmax_journal.py \
  --data-dir data/mnist \
  --epochs 25 \
  --batch-sizes 128 256 384 \
  --ironfi-alpha-by-batch 1 1.5 2.5 \
  --seeds 0 1 2 3 4 \
  --reg 1e-4 \
  --no-show
```

Outputs:
- `logs/mnist_softmax_journal_*/summary.json` (mean±std final accuracy + mean±std runtime)
- `figs/mnist_journal_train_loss_batch*.pdf`
- `figs/mnist_journal_test_acc_batch*.pdf`

### MNIST download note (SSL)

If your Python installation fails SSL certificate verification when downloading MNIST, you can set:

```bash
export IRONFI_ALLOW_INSECURE_SSL=1
```

This enables an insecure download fallback and should only be used on trusted networks.

---

## Nonconvex (JAX prototype) — for later

The JAX script (`experiments/nonconvex_iron_fi_jax.py`) mirrors the NumPy nonconvex experiment structure and outputs, but requires a Python ≥ 3.11 environment with `jax`/`jaxlib` (see `requirements-jax.txt`).

---

## 📄 License

This project is licensed under the **CC0 1.0 Universal** license - a public domain dedication that allows you to use, modify, and distribute this software freely for any purpose, including commercial use, without any restrictions.

**Key Points:**

- ✅ **Public Domain**: You can use this software for any purpose
- ✅ **No Attribution Required**: You don't need to credit the original authors
- ✅ **Commercial Use**: You can use it in commercial projects
- ✅ **Modification**: You can modify and distribute your changes
- ✅ **No Warranty**: The software is provided "as-is" without any warranties

**Why CC0?** This license promotes the ideal of a free culture and encourages the further production of creative, cultural, and scientific works by allowing maximum freedom of use and redistribution.

## 📧 Support and Contact

For questions, bug reports, or contributions, please contact:
**v dot leplat [at] innopolis dot ru**

