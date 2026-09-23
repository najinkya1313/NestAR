# NestAR

**Bayesian ARIMA model selection for astronomical time-series analysis using Nested Sampling.**

NestAR is a Python package for Bayesian model selection and inference with ARIMA models using Nested Sampling.

## What does NestAR do?

NestAR evaluates Bayesian evidences for candidate ARIMA models and provides posterior inference and forecasting for astronomical and other time-series data.

## Quick start


from nestar.ARIMA_ns import ARIMA_Nested_Sampler

model = ARIMA_Nested_Sampler(
    data=your_time_series,
    order=(2, 0, 1),
    mu_mean=0,
    mu_scale=1,
    num_live=500,
    num_delete=50,
    seed=42,
)

model.summary()