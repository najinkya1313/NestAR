# Quick Start

This guide demonstrates the basic workflow for fitting a single ARIMA model with NestAR.

## 1. Import NestAR

```python
from nestar.ARIMA_ns import ARIMA_Nested_Sampler
```

## 2. Prepare your time series

NestAR accepts an array-like one-dimensional time series.

For example:

```python
import numpy as np

data = np.loadtxt("your_time_series.txt")
```

Your data should contain the measurements in temporal order.

## 3. Fit an ARIMA model

Choose an ARIMA order `(p, d, q)` and specify the prior and nested-sampling settings:

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

Here, `(2, 0, 1)` corresponds to an ARIMA(2, 0, 1) model.

The nested sampler will run automatically when the `ARIMA_Nested_Sampler` object is created.

## 4. Inspect the results

Display the nested-sampling summary:

```python
model.summary()
```

This reports the nested-sampling runtime, posterior parameter means, and the estimated log Bayesian evidence.

The fitted posterior samples are available through:

```python
model.posterior_samples
```

The estimated log evidence and its uncertainty are available through:

```python
model.log_evidence
model.log_evidence_err
```

## 5. Obtain the posterior-mean fit

The fitted time series corresponding to the posterior mean parameters can be obtained with:

```python
y_fit = model.get_mean_forecasts()
```

To visualise the fit:

```python
model.mean_fit_plot(compare=True)
```

## 6. Forecasting

NestAR provides posterior-based in-sample and out-of-sample forecasting through:

```python
model.insample_forecast(...)
```

and:

```python
model.outsample_forecast(...)
```

These methods propagate posterior samples through the forecasting procedure and can be used to obtain predictive intervals.

See the [User Guide](user-guide.md) for details and examples of the forecasting interface.

## 7. Model comparison

For Bayesian comparison across multiple ARIMA orders, NestAR provides:

```python
from nestar.model_comparison_utils import ARIMA_model_comparison
```

A grid of candidate models can then be evaluated using:

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
).run()

results.plot_evidence_heatmap(max_p=5)
```

The resulting evidences can be used to compare the candidate ARIMA models.

For a complete worked example, see the notebooks in the [`examples/`](https://github.com/najinkya1313/NestAR/tree/main/examples) directory.
