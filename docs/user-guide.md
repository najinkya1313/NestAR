# User Guide

## Overview

NestAR provides a Bayesian framework for fitting and comparing ARIMA models using Nested Sampling.

A typical workflow is:

1. Prepare a time series.
2. Choose the differencing order \(d\).
3. Fit candidate ARIMA(p,d,q) models.
4. Compare their Bayesian evidences.
5. Analyse the posterior of the selected model.
6. Use the posterior samples for fitting and forecasting.

---

## ARIMA model specification

An ARIMA model is specified by the tuple:

```text
(p, d, q)
```

where:

* \(p\) is the autoregressive order.
* \(d\) is the differencing order.
* \(q\) is the moving-average order.

For example:

```python
order = (2, 0, 1)
```

corresponds to an ARIMA(2, 0, 1) model.

---

## Fitting a single model

The main class for fitting an individual ARIMA model is:

```python
from nestar.ARIMA_ns import ARIMA_Nested_Sampler
```

A model is fitted by creating an `ARIMA_Nested_Sampler` object:

```python
model = ARIMA_Nested_Sampler(
    data=data,
    order=(2, 0, 1),
    mu_mean=0,
    mu_scale=1,
    num_live=500,
    num_delete=50,
    seed=42,
)
```

The nested-sampling calculation is performed during initialization.

### Nested-sampling settings

The most important sampling parameters are:

| Parameter            | Description                                   |
| -------------------- | --------------------------------------------- |
| `num_live`           | Number of live points used by Nested Sampling |
| `num_delete`         | Number of points removed at each iteration    |
| `seed`               | Random seed                                   |
| `inner_steps_factor` | Controls the number of inner sampling steps   |
| `prior_type`         | Choice of prior parameterisation              |
| `prior_scale`        | Scale of the AR/MA coefficient priors         |

The nested-sampling calculation terminates when the estimated remaining contribution to the evidence satisfies the convergence criterion implemented by NestAR.

---

## Priors

NestAR provides multiple prior parameterisations for ARIMA coefficients.

### Constrained normal prior

The default prior uses normal distributions for the AR and MA coefficients while imposing stationarity and invertibility constraints.

The scale of the coefficient priors is controlled by:

```python
prior_scale
```

The long-term mean is controlled by:

```python
mu_mean
mu_scale
```

The noise scale is also assigned a prior.

### PACF-based prior

NestAR also provides a parameterisation in terms of partial autocorrelation-like variables constrained to the interval \((-1,1)\).

This option can be selected with:

```python
prior_type="pacf"
```

### Uniform prior

A uniform prior can be selected with:

```python
prior_type="uniform"
```

When using a uniform prior, bounds must be supplied through `prior_bounds`.

---

## Posterior analysis

After fitting, the nested-sampling posterior is available through:

```python
model.posterior_samples
```

The posterior mean of each fitted parameter is stored in:

```python
model.posterior_means
```

The estimated Bayesian evidence is available through:

```python
model.log_evidence
```

and its estimated uncertainty through:

```python
model.log_evidence_err
```

### Summary

Use:

```python
model.summary()
```

to display the sampling runtime, posterior parameter means, and a posterior corner plot.

---

## Posterior-mean fit

NestAR provides a fitted time series evaluated at the posterior-mean parameter values:

```python
y_fit = model.get_mean_forecasts()
```

The result can be visualised with:

```python
model.mean_fit_plot(compare=True)
```

---

## Model comparison

The model-comparison utilities allow multiple ARIMA orders to be evaluated.

Import the comparison routine with:

```python
from nestar.model_comparison_utils import ARIMA_model_comparison
```

A typical model-comparison calculation is:

```python
results = ARIMA_model_comparison(
    data=data,
    max_p=5,
    max_q=5,
    d=0,
    num_live=500,
    num_delete=50,
    seed=42,
    mu_mean=0,
    mu_scale=1,
)
```

For each candidate \((p,q)\) pair, NestAR performs an independent nested-sampling calculation.

The resulting evidences can be used to compare the candidate models.

---

## Forecasting

NestAR provides both in-sample and out-of-sample forecasting.

### In-sample forecasting

Use:

```python
model.insample_forecast(...)
```

to propagate the posterior through the model over the training data.

### Out-of-sample forecasting

Use:

```python
model.outsample_forecast(...)
```

to generate posterior-based predictions beyond the training interval.

Both methods can propagate multiple posterior samples and provide predictive uncertainty.

---

## Saved results

NestAR includes utilities for working with saved nested-sampling results without rerunning the original calculation.

The `arima_results` module provides interfaces for analysing saved posterior chains and evidence-comparison results:

```python
from nestar import arima_results
```

The two main result classes are:

```python
arima_results.PosteriorResults
```

and:

```python
arima_results.EvidenceResults
```

These interfaces are useful when the nested-sampling calculation has already been performed and only posterior analysis, plotting, or forecasting is required.

---

## Reproducible analyses

The repository contains the example notebooks used for the analyses in the associated paper.

The notebooks can be found in:

```text
examples/
```

and supporting result files are provided in:

```text
Heatmap Data/
NS Chains/
Plots/
```

See [Reproducing the Paper](reproduction.md) for details.
