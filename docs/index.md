![NestAR](assets/forecast_image.pdf)

## Nested sampling for ARIMA model selection

NestAR is a Python package for Bayesian model selection and inference with ARIMA (Autoregressive Integrated Moving Average) models using Nested Sampling.

## What does NestAR do?

It evaluates the Bayesian evidences for candidate ARIMA models and provides weighted posterior samples for inference, time series fitting and forecasting.

## Quick start

```python
from nestar.ARIMA_ns import ARIMA_Nested_Sampler
from nestar.ARIMA import ARIMA_forecast
import jax.numpy as jnp

your_time_series = ARIMA_forecast(
    jnp.array([2.0]),
    (1, 0, 1),
    1,
    1,
    [0.8],
    [0.4],
    1000,
    jnp.array([5.0]),
    30
)

model = ARIMA_Nested_Sampler(
    data=your_time_series,
    order=(1, 0, 1),
    mu_mean=0,
    mu_scale=1,
    num_live=500,
    num_delete=50,
    seed=42,
)

model.summary()
```

## Citation

If you use NestAR in your research or other work, please cite:

**Naik, A. J., & Handley, W. 2025**, *Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis*, arXiv:2512.01929. [doi:10.48550/arXiv.2512.01929](https://doi.org/10.48550/arXiv.2512.01929)

See the [full citation information](citation.md), including BibTeX.

