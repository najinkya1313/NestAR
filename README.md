<p align="center">
  <img src="nestar_logo.png" alt="NestAR banner" width="100%">
</p>

> **Bayesian ARIMA model selection for astronomical time-series analysis using Nested Sampling.**

[![arXiv](https://img.shields.io/badge/arXiv-2512.01929-b31b1b.svg)](https://arxiv.org/abs/2512.01929)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

NestAR combines **ARIMA models** with **Nested Sampling** for Bayesian model selection of astronomical time-series data.

Rather than selecting the ARIMA order using only maximum-likelihood criteria such as AIC or BIC, NestAR evaluates the **Bayesian evidence** for candidate ARIMA(p,d,q) models. The evidence naturally incorporates an Occam penalty, allowing increasingly complex models to be compared within a Bayesian framework.

For a selected model, NestAR also provides posterior samples and posterior-based time-series reconstruction and forecasting.

The sampler is built using [BlackJAX](https://github.com/blackjax-devs/blackjax), with [JAX](https://github.com/jax-ml/jax) providing the numerical backend.

**Paper**

> Naik, A. & Handley, W. (2025). *Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis.* [arXiv:2512.01929](https://arxiv.org/abs/2512.01929)

**Repository**

> [github.com/najinkya1313/NestAR](https://github.com/najinkya1313/NestAR)

---

## Key Features

* **Bayesian model selection** — calculate log-evidences across a user-defined grid of ARIMA$(p,d,q)$ models.
* **Occam penalty** — Bayesian evidence naturally penalises unnecessarily complex models.
* **Posterior inference** — obtain posterior samples and parameter estimates for individual ARIMA models.
* **Constrained priors** — enforce AR stationarity and MA invertibility through coefficient-root constraints.
* **JAX-backed computation** — numerical calculations are implemented using JAX and the nested sampler is provided by BlackJAX.
* **Forecasting** — generate posterior-based in-sample and out-of-sample forecasts.
* **Saved results** — nested-sampling chains and model-comparison results can be saved and subsequently analysed without rerunning the sampler.
* **Astronomical applications** — examples include sunspot time series, Kepler and TESS light curves, and quasar variability.

---

## Repository Structure

```text
NestAR/
├── nestar/
│   ├── __init__.py
│   ├── ARIMA.py
│   ├── ARIMA_ns.py
│   ├── priors.py
│   ├── arima_results.py
│   └── model_comparison_utils.py
│
├── Examples/
│   ├── sunspots_arima.ipynb
│   ├── kic_12008916.ipynb
│   ├── kepler_exoplanets.ipynb
│   ├── tess_data.ipynb
│   ├── simulated_data.ipynb
│   └── ...
│
├── Plots/
│   └── ...
│
├── Heatmap Data/
│   └── ...
│
├── NS Chains/
│   └── ...
│
├── pyproject.toml
├── LICENSE
├── README.md
└── .gitignore
```

### Source modules

| Module                             | Description                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `nestar/ARIMA.py`                  | Core JAX implementation of the ARIMA recursion and forecasting routines, including `ARIMA_fast` and `ARIMA_forecast`.          |
| `nestar/ARIMA_ns.py`               | Main nested-sampling implementation, including `ARIMA_Nested_Sampler`, likelihood functions, and prior-parameter construction. |
| `nestar/priors.py`                 | Prior-sampling routines, including constrained normal priors and PACF-based priors.                                            |
| `nestar/model_comparison_utils.py` | Utilities for running and analysing comparisons over grids of ARIMA orders, together with stationarity-related utilities.      |
| `nestar/arima_results.py`          | Loading, analysing, plotting, and forecasting from saved ARIMA nested-sampling results and posterior chains.                   |

The `Examples/` directory contains notebooks demonstrating the use of NestAR on simulated and astronomical time-series data. The `Plots/`, `Heatmap Data/`, and `NS Chains/` directories contain figures and saved numerical results associated with the analyses.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/najinkya1313/NestAR.git
cd NestAR
```

Install NestAR and its declared dependencies:

```bash
python -m pip install .
```

For development, an editable installation can be used instead:

```bash
python -m pip install -e .
```

After installation, NestAR can be imported as a standard Python package:

```python
from nestar.ARIMA_ns import ARIMA_Nested_Sampler
```

For GPU or other accelerated JAX backends, install the appropriate JAX distribution following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

---

## Quick Start

### Fit a single ARIMA model

```python
from nestar.ARIMA_ns import ARIMA_Nested_Sampler

model = ARIMA_Nested_Sampler(
    data       = your_time_series,
    order      = (2, 0, 1),
    mu_mean    = 0,
    mu_scale   = 1,
    num_live   = 500,
    num_delete = 50,
    seed       = 42,
)

model.summary()

y_fit = model.get_mean_forecasts()

model.mean_fit_plot(compare=True)
```

### Run a model-comparison grid

```python
from nestar.model_comparison_utils import ARIMA_model_comparison

results = ARIMA_model_comparison(
    data       = your_time_series,
    max_p      = 5,
    max_q      = 5,
    d          = 0,
    num_live   = 500,
    num_delete = 50,
    seed       = 42,
    mu_mean    = 0,
    mu_scale   = 1,
    file_name  = "results.txt",
)
```

The exact return values and available plotting utilities depend on the selected model-comparison interface; see the example notebooks and API documentation for details.

### Analyse saved results

Results generated by a nested-sampling run can be loaded and analysed using the classes and utilities in:

```python
from nestar import arima_results
```

For example, posterior-chain analysis and forecasting are handled by the `PosteriorResults` interface.

---

## Method

### ARIMA Models

An ARIMA$(p,d,q)$ model is specified by three integer orders:

| Symbol | Meaning                                                   |
| ------ | --------------------------------------------------------- |
| **p**  | Autoregressive order — dependence on previous values      |
| **d**  | Differencing order                                        |
| **q**  | Moving-average order — dependence on previous innovations |

For a given value of $d$, NestAR can evaluate a grid of $(p,q)$ combinations and compare their Bayesian evidences.

### Priors

NestAR provides several prior parameterisations. The constrained normal-prior implementation samples AR and MA coefficients from normal distributions while enforcing:

* **stationarity** of the AR component, and
* **invertibility** of the MA component.

The constraints are imposed through the roots of the corresponding characteristic polynomials.

NestAR also provides a PACF-based parameterisation in which autoregressive and moving-average coefficients are constructed from parameters constrained to the interval $(-1,1)$.

### Nested Sampling

Nested Sampling computes the Bayesian evidence

$$
Z = \int \mathcal{L}(\theta)\,\pi(\theta)\,d\theta,
$$

by progressively contracting the prior volume around regions of increasing likelihood.

NestAR uses the nested sampler provided by BlackJAX. The implementation terminates according to the convergence criterion

```text
logZ_live - logZ < -3
```

For a set of candidate ARIMA models, the resulting evidences can be compared directly or combined with prior probabilities over the model orders.

---

## Forecasting

A fitted `ARIMA_Nested_Sampler` object provides posterior-based time-series analysis, including:

```python
model.mean_fit_plot()
```

for a posterior-mean fit, as well as:

```python
model.insample_forecast(...)
```

and

```python
model.outsample_forecast(...)
```

for posterior-based in-sample and out-of-sample forecasting.

The forecasting utilities in `arima_results.py` can also operate directly on saved posterior chains.

---

## Examples and Reproducing the Paper

The `Examples/` directory contains the Python notebooks used for the analyses presented in the paper.

Examples include:

* **Sunspots**
* **KIC 12008916**
* **Kepler-17**
* **TESS data**
* **Simulated ARIMA data**
* **Other astronomical time series**

The repository also contains the saved evidence calculations, nested-sampling chains, and publication figures used in the analyses.

For a complete reproduction of the paper, see the notebooks in `Examples/` together with the corresponding data in:

```text
Heatmap Data/
NS Chains/
Plots/
```

Before running the notebooks, install NestAR from the repository root:

```bash
python -m pip install -e .
```

---

## API Overview

### `ARIMA_fast`

Located in:

```python
nestar.ARIMA
```

`ARIMA_fast` provides the JAX-compiled ARIMA recursion used by the likelihood and fitting routines.

### `ARIMA_forecast`

Located in:

```python
nestar.ARIMA
```

`ARIMA_forecast` generates out-of-sample forecasts for a specified ARIMA model.

### `ARIMA_Nested_Sampler`

Located in:

```python
nestar.ARIMA_ns
```

`ARIMA_Nested_Sampler` performs nested sampling for a fixed ARIMA$(p,d,q)$ order.

Important arguments include:

| Argument             | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `data`               | Input time series                                      |
| `order`              | ARIMA order `(p, d, q)`                                |
| `mu_mean`            | Prior mean for the long-term mean                      |
| `mu_scale`           | Prior scale for the long-term mean                     |
| `num_live`           | Number of live points                                  |
| `num_delete`         | Number of points removed per iteration                 |
| `seed`               | Random seed                                            |
| `prior_scale`        | Scale of AR/MA coefficient priors                      |
| `inner_steps_factor` | Number of inner sampling steps per parameter dimension |
| `prior_type`         | Prior parameterisation                                 |
| `prior_bounds`       | Bounds for uniform priors                              |

Important attributes include:

```python
model.log_evidence
model.log_evidence_err
model.posterior_samples
model.posterior_means
```

Important methods include:

```python
model.summary()
model.get_mean_forecasts()
model.mean_fit_plot()
model.insample_forecast(...)
model.outsample_forecast(...)
```

### `ARIMA_model_comparison`

Located in:

```python
nestar.model_comparison_utils
```

Runs nested-sampling fits over a grid of candidate ARIMA orders for model comparison.

### `PosteriorResults`

Located in:

```python
nestar.arima_results
```

Provides analysis and forecasting tools for posterior chains saved from a nested-sampling run.

### `EvidenceResults`

Located in:

```python
nestar.arima_results
```

Provides an interface for analysing saved model-comparison/evidence results.

---

## Citation

If you use NestAR in your research, please cite:

```bibtex
@article{naik2025nestar,
  title   = {Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis},
  author  = {Naik, Ajinkya and Handley, Will},
  journal = {arXiv preprint arXiv:2512.01929},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.01929}
}
```

A versioned Zenodo archive of the software should also be cited when available.

---

## Acknowledgements

This work was carried out in the research environment of the [Handley Lab](https://handley-lab.co.uk/) at the **Institute of Astronomy, University of Cambridge**.

NestAR uses the following open-source software:

* [JAX](https://github.com/jax-ml/jax)
* [BlackJAX](https://github.com/blackjax-devs/blackjax)
* [anesthetic](https://github.com/handley-lab/anesthetic)
* [fgivenx](https://github.com/handley-lab/fgivenx)

---

## License

NestAR is released under the MIT License. See [LICENSE](LICENSE) for details.
