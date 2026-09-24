# Installation

## Requirements

NestAR requires Python 3.10 or later.

The package uses [JAX](https://docs.jax.dev/) for numerical computation and [BlackJAX](https://github.com/blackjax-devs/blackjax) for nested sampling. NestAR currently uses a specific historical BlackJAX version because later releases changed the nested-sampling API used by the package. Additional dependencies are installed automatically when NestAR is installed.

## Install from GitHub

Clone the NestAR repository:

```bash
git clone https://github.com/najinkya1313/NestAR.git
cd NestAR
```

Install NestAR and its dependencies:

```bash
python -m pip install .
```

For Jupyter notebook functionality, including the progress bars used by some plotting routines, install the optional notebook dependencies:

```bash
python -m pip install "nestar[notebook]"
```

For development, an editable installation can be used:

```bash
python -m pip install -e .
```

With an editable installation, changes made to the source code are immediately reflected when NestAR is imported, without requiring the package to be reinstalled.

## Verify the installation

After installation, verify that NestAR can be imported:

```bash
python -c "from nestar.ARIMA_ns import ARIMA_Nested_Sampler; print('NestAR installation successful')"
```

A successful installation should produce:

```text
NestAR installation successful
```

## JAX and hardware acceleration

NestAR uses JAX as its numerical backend and BlackJAX for nested sampling.

The default installation is suitable for CPU-based computation. Users wishing to run JAX with a GPU or another accelerated backend should follow the appropriate instructions in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Running the examples

The example notebooks accompanying the paper are provided in the `examples/` directory.

After installing NestAR and, for notebook use, the optional notebook dependencies, launch Jupyter from the repository root:

```bash
jupyter notebook
```

Then open the relevant notebook from the `examples/` directory.

Running Jupyter from the repository root is recommended so that the paths to the supplied data and output files are resolved consistently.