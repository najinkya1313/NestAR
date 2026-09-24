<p align="center">
  <img src="nestar_logo.png" alt="NestAR banner" width="100%">
</p>

> **Bayesian ARIMA model selection for astronomical time-series analysis using Nested Sampling.**

[![arXiv](https://img.shields.io/badge/arXiv-2512.01929-b31b1b.svg)](https://arxiv.org/abs/2512.01929)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## Overview

**NestAR** combines **Nested Sampling** with **ARIMA** (**A**uto**R**egressive **I**ntegrated **M**oving **A**verage) models for Bayesian model selection. 

It provides a rigorous alternative to maximum-likelihood based selection and fitting of ARIMA models, such as using the Bayesian or Akaike Information Criterion.

For a selected model, NestAR also provides posterior samples, model reconstruction, and posterior-based forecasting.

The nested-sampling implementation uses [BlackJAX](https://github.com/blackjax-devs/blackjax), with [JAX](https://github.com/jax-ml/jax) providing the numerical backend.

## Installation

Clone the repository:

```bash
git clone https://github.com/najinkya1313/NestAR.git
cd NestAR
```

Install NestAR and its dependencies:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

## Quick Start

Fit a single ARIMA model using Nested Sampling:

```python
from nestar.ARIMA_ns import ARIMA_Nested_Sampler

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


## Documentation

Full documentation, including the API reference, examples, and detailed usage information, is available at:

**[NestAR Documentation](https://nestar.readthedocs.io/)**

## Citation

If you use NestAR in your research, please cite the accompanying paper:

> A. J. Naik & W. Handley, *Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis*.

[arXiv:2512.01929](https://arxiv.org/abs/2512.01929)

```bibtex
@article{naik2025nestar,
  title         = {Nested Sampling for ARIMA Model Selection in Astronomical Time-Series Analysis},
  author        = {Naik, Ajinkya and Handley, Will},
  journal       = {arXiv preprint arXiv:2512.01929},
  year          = {2025},
  eprint        = {2512.01929},
  archivePrefix = {arXiv}
}
```

## License

NestAR is released under the MIT License. See [LICENSE](LICENSE) for details.
