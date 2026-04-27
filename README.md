# IRON-FI: Implicit Resolvent Optimization under Noise

This repository contains the Python implementation of **IRON-FI** (fully implicit resolvent / Backward–Euler discretization) and the experimental suite used in the IRON preprint.

Core IRON-FI steps:
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

If you hit an import error like `ModuleNotFoundError: No module named 'plots'`, it usually means
the editable install step was skipped or not run from the repo root. Re-run:

```bash
pip install -e .
```

---

## Project layout

```
datasets/
  mnist.py                     # download/cache MNIST into data/mnist/ and load NumPy arrays
experiments/
  quad_iron_fi.py              # quadratic IRON-FI + stationary MSE / Lyapunov validation
  nonconvex_iron_fi_numpy.py   # qualitative nonconvex log-cosh IRON-FI (NumPy)
  nonconvex_iron_fi_jax.py     # nonconvex log-cosh IRON-FI (JAX prototype)
  logreg_synth_ironfi.py       # Part A: synthetic ridge-logistic regression (IRON-only, multi-seed)
  mnist_softmax_benchmark.py   # Part B: single-run exploratory MNIST benchmark
  mnist_softmax_journal.py     # Part B: validation-tuned multi-seed MNIST runner
  mnist_softmax_core.py        # shared train/val/test training loops for Part B runners
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

This script now serves two purposes:

- Validate the theorem-facing quadratic quantity
  $$\widehat{\mathrm{MSE}}_k= \frac{1}{N} \sum_{j=1}^{N}\left\|x_k^{(j)}-x^\star\right\|^2$$

  together with its bias--variance decomposition
  $$ \widehat{\mathrm{MSE}}_k = \left\|\bar x_k-x^\star\right\|^2 + \mathrm{tr}\!\left(\widehat{\mathrm{Cov}}(x_k)\right).$$

- Check the stationary quadratic prediction by plotting
  $$ \alpha\,\widehat{\mathrm{MSE}}_\infty$$

  against $\alpha$, and comparing it with the exact discrete-Lyapunov stationary MSE and the asymptotic constant
  $$C_{\mathrm{quad}}.$$

Running

```bash
python experiments/quad_iron_fi.py \
  --alpha-scale 1 10 200 500 \
  --nsamples 20000 --iters 100 --sigma 1.0 --seed 0 \
  --eigs 1.0 1.0 3.0 \
  --save-figs --no-show
```

generates, for each $\alpha$:
- `figs/quad_mean_alpha{α}.pdf`
- `figs/quad_clouds_alpha{α}.pdf`

and also:
- `figs/quad_stationary_scaled_mse.pdf`
- `logs/quad_iron_fi_summary_seed<seed>.json`

Interpretation of the outputs:
- `quad_mean_alpha{α}.pdf`: ensemble MSE, squared bias, covariance trace, and mean error over iterations.
- `quad_clouds_alpha{α}.pdf`: qualitative projected particle clouds (initial vs late-time final state).
- `quad_stationary_scaled_mse.pdf`: empirical $\alpha\,\widehat{\mathrm{MSE}}_\infty$, exact Lyapunov $\alpha\,\mathrm{MSE}_\infty$, and the asymptotic constant $C_{\mathrm{quad}}$.

Notes:
- The stationary-constant validation is done in the fixed-$\gamma$ setting (`--gamma-mode fixed`), which is the default and matches the exact Lyapunov formula used in the script.
- The cloud figures are intentionally qualitative; the main quantitative comparison to theory is carried by `quad_mean_alpha*.pdf` and `quad_stationary_scaled_mse.pdf`.

---

## Nonconvex experiment (NumPy, log-cosh)

This script is intended as a **qualitative / illustrative** nonconvex experiment, not as a theorem-validation figure.
It now distinguishes:
- a last-iterate cloud,
- a pooled late-time cloud after burn-in,
- and objective-slice overlays built from the pooled late-time samples.

Running

```bash
python experiments/nonconvex_iron_fi_numpy.py \
  --alpha-scale 1 10 200 500 \
  --nsamples 20000 --iters 10 --sigma 1.0 --seed 0 \
  --newton-it 6 --tol 1e-8 \
  --step-cap 0.5 --max-ls 12 --clip-x 30 \
  --plot-lim 5 \
  --save-figs --no-show
```

generates (per $\alpha$):
- `figs/ncx_numpy_alpha{α}_mean_norm.pdf`
- `figs/ncx_numpy_alpha{α}_cloud.pdf`
- `figs/logcosh_slices_alpha{α}.pdf`

and:
- `logs/nonconvex_alpha{α}_seed<seed>.json`

Interpretation of the outputs:
- `ncx_numpy_alpha{α}_mean_norm.pdf`: mean-iterate norm proxy only; this is not a theorem-level quantity.
- `ncx_numpy_alpha{α}_cloud.pdf`: top row = last-iterate projected clouds, bottom row = pooled late-time projected clouds after burn-in.
- `logcosh_slices_alpha{α}.pdf`: objective slices with projected late-time clouds.

Notes:
- The default `--burn-frac 0.5` means the pooled cloud uses the last half of the run.
- The default `--gamma-mode updated` matches the toy experiment narrative used elsewhere in the repository.
- If the contour layers near minima are visually too compressed, adjust the contour scaling in `plots/slices.py`; the script is designed for qualitative inspection rather than exact quantitative matching.

---

## Synthetic ridge-logistic regression (IRON-only)

This suite validates:
- stationary MSE scaling $\widehat{\mathrm{MSE}}(\alpha)\sim 1/\alpha$ (slope close to \(-1\) on log–log),
- tolerance sweep showing $\varepsilon$ does not need to shrink with $\alpha$,
- mean inner LM/Newton iterations vs $\alpha$,
- confidence intervals on the fitted slope across seeds,
- scaled-MSE diagnostics to show when a tolerance breaks the \(1/\alpha\) trend.

```bash
python experiments/logreg_synth_ironfi.py \
  --n 20000 --d 50 --iters 1000 --burn-frac 0.3 \
  --alpha-grid 1 2 5 10 20 50 100 200 \
  --tol-grid 1e-2 1e-4 1e-6 \
  --seeds 0 1 2 3 4 \
  --sigma 1.0 --reg 1e-2 --seed 0 \
  --slope-fit-min-alpha 5 \
  --no-show
```

Outputs:
- `figs/synth_logreg_mse_vs_alpha_tol<best>.pdf`
- `figs/synth_logreg_tol_effect.pdf`
- `figs/synth_logreg_inner_iters_vs_alpha.pdf`
- `figs/synth_logreg_scaled_mse_vs_alpha.pdf`
- `logs/logreg_synth_ironfi_*/summary.json`
- `logs/logreg_synth_ironfi_*/slopes.json`

---

## MNIST softmax regression benchmark (IRON-FI vs NAG-GS vs AdamW)

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

This script is mainly a quick exploratory runner. For paper-facing figures, prefer the journal script below.

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

### Grid search (10 epochs) for IRON-FI $\alpha$

This quick benchmark now tunes IRON-FI on a validation split rather than on the test set.

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

This is the paper-facing MNIST runner. It assumes MNIST is already cached locally (see the download step above).

```bash
python experiments/mnist_softmax_journal.py \
  --data-dir data/mnist \
  --epochs 25 \
  --tune-epochs 10 \
  --batch-sizes 128 256 384 \
  --seeds 0 1 2 3 4 \
  --reg 1e-4 \
  --no-show
```

Outputs:
- `logs/mnist_softmax_journal_*/summary.json` (selected hyperparameters, tuning records, final accuracy, runtime)
- `figs/mnist_journal_train_loss_batch*.pdf`
- `figs/mnist_journal_test_acc_batch*.pdf`
- `figs/mnist_journal_test_acc_vs_time_batch*.pdf`
- `figs/mnist_journal_ironfi_inner_batch*.pdf`

Notes:
- The journal script uses a train/validation/test protocol: tuning is done on validation, and test is reported only in the final evaluation.
- AdamW, NAG-GS, and IRON-FI are tuned under the same selection rule (final validation accuracy after `--tune-epochs`).
- If you do not pass explicit tuning grids, the script uses its built-in default grids for AdamW, NAG-GS, and IRON-FI.
- The time-based accuracy plots are the main runtime-sensitive comparison and complement the epoch-based plots.

### MNIST download note (SSL)

If your Python installation fails SSL certificate verification when downloading MNIST, you can set:

```bash
export IRONFI_ALLOW_INSECURE_SSL=1
```

This enables an insecure download fallback and should only be used on trusted networks.

If the MNIST download looks stuck (no output for a long time), enable verbose download logs and/or increase the network timeout:

```bash
export IRONFI_MNIST_VERBOSE=1
export IRONFI_MNIST_TIMEOUT_S=120
```

---

## Nonconvex (JAX prototype) — for later

The JAX script (`experiments/nonconvex_iron_fi_jax.py`) mirrors the NumPy nonconvex experiment structure and outputs, but requires a Python ≥ 3.11 environment with `jax`/`jaxlib` (see `requirements-jax.txt`).

---

## 📄 License

This project is licensed under the **MIT License** (see `LICENSE`).

**Key points (MIT):**

- ✅ **Use**: you can use this software for any purpose
- ✅ **Modify & distribute**: you can modify, distribute, and sublicense it
- ✅ **Commercial use**: permitted
- ✅ **Attribution**: include the copyright and license notice in copies
- ✅ **No warranty**: the software is provided "as is"

## 📧 Support and Contact

For questions, bug reports, or contributions, please contact:
**valentin dot leplat [at] gmail dot com**

