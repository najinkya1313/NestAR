"""
arima_mle.py

Independent multi-start maximum-likelihood optimisation for ARIMA models.

- Supports loglikelihood and pacf_loglikelihood.
- Uses a tanh/PACF parameterisation to enforce stationarity/invertibility.
- Starting points are random by default. Optionally, pre-computed starting
  points (e.g. the highest-likelihood points of a nested-sampling chain) can
  be supplied via ``start_points``; they are then optimised in addition to
  any random starts.
- Optionally reads an existing evidence file and uses its MaxLogL values
  as lower bounds.
- Recalculates BIC and updates the evidence file.
- Does not require nested-sampling chains.
"""

import ast
import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from ARIMA_ns import loglikelihood, pacf_loglikelihood, pacf_to_arma
import arima_results as ar


# ------------------------------------------------------------
# Parameter construction
# ------------------------------------------------------------

def _build_params(x, order, likelihood_type):
    """
    Convert an unconstrained optimiser vector into likelihood parameters.

    x : unconstrained parameter vector, laid out as
        [arctanh(PACF_AR), arctanh(PACF_MA), log(sigma), mu, init_y...]
    order : (p, d, q) ARIMA order
    likelihood_type : 'normal' (phi/theta parameterisation) or 'pacf'
        (PACF parameterisation)

    Returns a dict of named parameters in the format expected by
    ``loglikelihood`` or ``pacf_loglikelihood``. Stationarity and
    invertibility are enforced by the tanh/PACF transform.
    """
    p, d, q = order

    alpha_ar = jnp.tanh(x[:p])
    alpha_ma = jnp.tanh(x[p:p+q])

    sigma = jnp.exp(x[p+q])
    mu = x[p+q+1]
    init_y = x[p+q+2:]

    params = {
        'sigma': sigma,
        'mu': mu
    }

    if likelihood_type == 'pacf':

        for i in range(p):
            params[f'alpha_ar_{i+1}'] = alpha_ar[i]

        for j in range(q):
            params[f'alpha_ma_{j+1}'] = alpha_ma[j]

    elif likelihood_type == 'normal':

        phi = pacf_to_arma(alpha_ar) if p else jnp.array([])
        theta = -pacf_to_arma(alpha_ma) if q else jnp.array([])

        for i in range(p):
            params[f'phi_{i+1}'] = phi[i]

        for j in range(q):
            params[f'theta_{j+1}'] = theta[j]

    else:
        raise ValueError(
            "likelihood_type must be 'normal' or 'pacf'."
        )

    for i in range(p):
        params[f'init_y_{i+1}'] = init_y[i]

    return params


# ------------------------------------------------------------
# Converting external samples (e.g. NS chains) to start vectors
# ------------------------------------------------------------

def arma_to_pacf(coefs):
    """
    Inverse of ``pacf_to_arma`` (standard Durbin-Levinson step-down).

    coefs : array of AR (or negated MA) coefficients

    Returns the partial autocorrelation coefficients, same length as coefs.
    Values with |alpha| >= 1 indicate a non-stationary / non-invertible input.
    """
    coefs = np.asarray(coefs, dtype=float).copy()
    k = len(coefs)
    alpha = np.zeros(k)
    for m in range(k - 1, -1, -1):
        a = coefs[m]
        alpha[m] = a
        if m > 0:
            prev = coefs[:m]
            coefs = (prev + a * prev[::-1]) / (1.0 - a**2)
    return alpha


def samples_to_unconstrained(samples, idx, order, prior_type):
    """
    Build optimiser start vectors from rows of a nested-sampling
    posterior-samples object.

    samples : posterior samples object indexable by parameter name
        (e.g. ``model.posterior_samples``)
    idx : integer indices of the rows (samples) to convert
    order : (p, d, q) ARIMA order
    prior_type : 'pacf' if the samples hold alpha_ar_*/alpha_ma_*,
        otherwise ('normal' or 'uniform') phi_*/theta_* are assumed

    Returns an array of shape (n_good, n_params) with rows
    [arctanh(alpha_ar), arctanh(alpha_ma), log(sigma), mu, init_y...],
    i.e. the layout used by ``_build_params``. Rows that cannot be mapped
    (non-stationary, non-invertible, or failing the round-trip check against
    ``pacf_to_arma``) are dropped, and duplicate rows are removed.
    """
    p, d, q = order
    col = lambda k: np.asarray(samples[k], dtype=float)[idx]

    sigma, mu = col('sigma'), col('mu')
    init_y = (np.stack([col(f'init_y_{i+1}') for i in range(p)], axis=1)
              if p else np.zeros((len(idx), 0)))

    if prior_type == 'pacf':
        a_ar = (np.stack([col(f'alpha_ar_{i+1}') for i in range(p)], axis=1)
                if p else np.zeros((len(idx), 0)))
        a_ma = (np.stack([col(f'alpha_ma_{j+1}') for j in range(q)], axis=1)
                if q else np.zeros((len(idx), 0)))
    else:
        phi = (np.stack([col(f'phi_{i+1}') for i in range(p)], axis=1)
               if p else np.zeros((len(idx), 0)))
        theta = (np.stack([col(f'theta_{j+1}') for j in range(q)], axis=1)
                 if q else np.zeros((len(idx), 0)))
        a_ar = np.array([arma_to_pacf(r) for r in phi]).reshape(len(idx), p)
        a_ma = np.array([arma_to_pacf(-r) for r in theta]).reshape(len(idx), q)

        # sanity check against the actual pacf_to_arma convention
        for a, ref, sign in [(a_ar, phi, 1.0), (a_ma, theta, -1.0)]:
            for r in range(len(idx)):
                if a.shape[1] and np.all(np.abs(a[r]) < 1):
                    back = sign * np.asarray(pacf_to_arma(jnp.asarray(a[r])))
                    if not np.allclose(back, ref[r], atol=1e-3):
                        a[r] = np.nan   # flag as bad

    alpha = np.concatenate([a_ar, a_ma], axis=1)
    x = np.concatenate([
        np.arctanh(np.clip(alpha, -1 + 1e-6, 1 - 1e-6)),
        np.log(sigma)[:, None], mu[:, None], init_y
    ], axis=1)

    good = (np.all(np.isfinite(alpha), axis=1)
            & np.all(np.abs(alpha) < 1, axis=1)
            & np.all(np.isfinite(x), axis=1))
    x = x[good]
    return np.unique(x, axis=0) if len(x) else x


# ------------------------------------------------------------
# Random starting points
# ------------------------------------------------------------

def _random_starts(data, order, neg_loglik, n_random=1000,
                   top_k=10, rng_seed=0):
    """
    Draw random starting points and keep the best few.

    data : time series data
    order : (p, d, q) ARIMA order
    neg_loglik : function mapping an unconstrained vector to the negative
        log-likelihood
    n_random : number of random starting points to draw and evaluate
    top_k : number of best (lowest negative log-likelihood) points to keep
    rng_seed : seed of the random number generator

    Returns (starts, values): the top_k unconstrained start vectors and their
    negative log-likelihood values.
    """

    p, d, q = order

    data = np.asarray(data, dtype=float)
    diff_data = np.diff(data, n=d) if d else data

    mu0 = float(np.mean(diff_data))
    sigma0 = max(float(np.std(diff_data)), 1e-8)

    rng = np.random.default_rng(rng_seed)

    # Generate PACF values inside (-1, 1)
    alpha = rng.uniform(
        -0.95, 0.95, size=(n_random, p + q)
    )

    X = np.concatenate([
        np.arctanh(alpha),
        np.full((n_random, 1), np.log(sigma0)),
        np.full((n_random, 1), mu0),
        np.full((n_random, p), mu0)
    ], axis=1)

    # Evaluate all starting points
    evaluate = jax.jit(jax.vmap(neg_loglik))

    values = np.asarray(
        evaluate(jnp.asarray(X, dtype=jnp.float32))
    )

    values = np.where(np.isfinite(values), values, np.inf)

    indices = np.argsort(values)[:top_k]

    return X[indices], values[indices]


# ------------------------------------------------------------
# BFGS optimisation
# ------------------------------------------------------------

def _optimise(neg_loglik, x0, maxiter=500):
    """
    Run BFGS from a single starting point using JAX gradients.

    neg_loglik : function mapping an unconstrained vector to the negative
        log-likelihood
    x0 : unconstrained starting vector
    maxiter : maximum number of BFGS iterations

    Returns the ``scipy.optimize.OptimizeResult``. Non-finite values or
    gradients are replaced by a large penalty so BFGS does not crash.
    """

    value_and_grad = jax.jit(
        jax.value_and_grad(neg_loglik)
    )

    def objective(x):

        value, gradient = value_and_grad(
            jnp.asarray(x, dtype=jnp.float32)
        )

        value = float(value)
        gradient = np.asarray(gradient, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return 1e30, np.zeros_like(x)

        return value, gradient

    return minimize(
        objective,
        x0,
        jac=True,
        method='BFGS',
        options={'maxiter': maxiter}
    )


# ------------------------------------------------------------
# Optimise one ARIMA order
# ------------------------------------------------------------

def mle_for_order(
    data,
    order,
    likelihood_type='normal',
    floor_logL=-np.inf,
    seed=0,
    meas_sigma=None,
    n_random=1000,
    top_k=10,
    maxiter=500,
    rng_seed=0,
    start_points=None
):
    """
    Find the maximum likelihood for one ARIMA order using multi-start BFGS.

    data : time series data to be analyzed
    order : (p, d, q) ARIMA order
    likelihood_type : 'normal' uses loglikelihood (phi/theta parameters),
        'pacf' uses pacf_loglikelihood (PACF parameters)
    floor_logL : existing (e.g. NS) maximum log-likelihood. The returned
        value cannot be lower than this value.
    seed : random seed passed to the likelihood function
    meas_sigma : per-point measurement uncertainties, or None
    n_random : number of random starting points drawn and ranked. Set to 0
        to use only ``start_points``.
    top_k : number of best random starting points that are optimised
    maxiter : maximum BFGS iterations per starting point
    rng_seed : seed of the random starting-point generator
    start_points : optional array of shape (n, n_params) of additional
        starting points in the unconstrained parametrisation (see
        ``samples_to_unconstrained``), e.g. the highest-likelihood points of
        a nested-sampling chain. If None (default), only random starting
        points are used. All supplied points are optimised, in addition to
        the top_k random ones.

    Returns a dict with keys 'logL_mle', 'gain' (logL_mle - floor_logL),
    'successful_runs', 'n_starts' and 'best_result'.
    """

    p, d, q = order

    if likelihood_type == 'normal':
        llk = loglikelihood(
            data, order, seed, meas_sigma=meas_sigma
        )

    elif likelihood_type == 'pacf':
        llk = pacf_loglikelihood(
            data, order, seed, meas_sigma=meas_sigma
        )

    else:
        raise ValueError(
            "likelihood_type must be 'normal' or 'pacf'."
        )

    neg_loglik = lambda x: -llk(
        _build_params(x, order, likelihood_type)
    )

    start_list, value_list = [], []

    # Generate and rank independent random starting points
    if n_random > 0:
        s, v = _random_starts(
            data,
            order,
            neg_loglik,
            n_random=n_random,
            top_k=top_k,
            rng_seed=rng_seed
        )
        start_list.append(s)
        value_list.append(v)

    # Optionally add externally supplied starting points
    if start_points is not None and len(start_points) > 0:
        sp = np.asarray(start_points, dtype=float)
        v = np.asarray(
            jax.jit(jax.vmap(neg_loglik))(
                jnp.asarray(sp, dtype=jnp.float32)
            )
        )
        start_list.append(sp)
        value_list.append(np.where(np.isfinite(v), v, np.inf))

    if not start_list:
        raise ValueError(
            "Need n_random > 0 or a non-empty start_points."
        )

    starts = np.concatenate(start_list, axis=0)
    start_values = np.concatenate(value_list, axis=0)

    best_logL = max(
        float(floor_logL),
        float(-np.min(start_values))
    )

    best_result = None
    successful = 0

    # Optimise from each selected starting point
    for i, x0 in enumerate(starts):

        result = _optimise(
            neg_loglik,
            x0,
            maxiter=maxiter
        )

        if result.success:
            successful += 1

        candidate = -float(result.fun)

        if np.isfinite(candidate) and candidate > best_logL:
            best_logL = candidate
            best_result = result

    return {
        'logL_mle': best_logL,
        'gain': best_logL - floor_logL,
        'successful_runs': successful,
        'n_starts': len(starts),
        'best_result': best_result
    }


# ------------------------------------------------------------
# Optimise entire grid and optionally update evidence file
# ------------------------------------------------------------

def run_mle_bic(
    data,
    orders=None,
    evidence_file=None,
    out_file=None,
    likelihood_type='normal',
    seed=0,
    meas_sigma=None,
    include_mean=True,
    include_scale=True,
    n_random=1000,
    top_k=10,
    maxiter=500,
    rng_seed=0,
):
    """
    Optimise the maximum likelihood for each ARIMA order of a grid and
    recompute BIC.

    data : time series data to be analyzed
    orders : list of (p, d, q) orders to optimise. If None, all orders in
        ``evidence_file`` are used.
    evidence_file : if given, existing evidence file whose MaxLogL values are
        used as lower bounds. The existing evidence and other quantities are
        preserved; MaxLogL and BIC are replaced and NSMaxLogL, MLEGain and
        MLESuccessfulRuns are added.
    out_file : output path for the updated evidence file. Defaults to the
        evidence file name with '_mle' appended.
    likelihood_type : 'normal' or 'pacf', see ``mle_for_order``
    seed : random seed passed to the likelihood function
    meas_sigma : per-point measurement uncertainties, or None
    include_mean, include_scale : whether the BIC parameter count includes a
        mean and/or noise-scale parameter
    n_random : number of random starting points per order
    top_k : number of best random starting points optimised per order
    maxiter : maximum BFGS iterations per starting point
    rng_seed : base seed of the random starting-point generator (offset by
        the order index)

    If evidence_file is None, ``orders`` must be supplied and the
    optimisation is performed without a nested-sampling lower bound.

    Returns a list of result dicts, one per order (see ``mle_for_order``,
    plus 'order', 'NSMaxLogL' and 'BIC').
    """

    data = np.asarray(data, dtype=float)
    n_data = len(data)

    records = []

    # Read evidence file, if supplied
    if evidence_file is not None:

        with open(evidence_file, 'r') as f:
            records = [
                ar._parse_line(line.strip())
                for line in f if line.strip()
            ]

        file_orders = [
            ast.literal_eval(fields['Order'])
            for fields in records
        ]

        if orders is None:
            orders = file_orders

        elif not set(orders).issubset(set(file_orders)):
            raise ValueError(
                "Some requested orders are absent from the evidence file."
            )

        if out_file is None:
            root, ext = os.path.splitext(evidence_file)
            out_file = root + '_mle' + ext

    elif orders is None:
        raise ValueError(
            "Supply either an evidence_file or an explicit list of orders."
        )

    

    results = []
    output_records = []

    for index, order in enumerate(orders):

        print(f"\nOptimising ARIMA{order}...")

        # Find the corresponding NS maximum, if available
        floor_logL = -np.inf
        fields = None

        if evidence_file is not None:
            fields = next(
                f for f in records
                if ast.literal_eval(f['Order']) == order
            )

            floor_logL = float(fields['MaxLogL'])

            # Preserve the original NS maximum across repeated runs
            ns_ref = float(
                fields.get('NSMaxLogL', floor_logL)
            )

        else:
            ns_ref = np.nan

        result = mle_for_order(
            data=data,
            order=order,
            likelihood_type=likelihood_type,
            floor_logL=floor_logL,
            seed=seed,
            meas_sigma=meas_sigma,
            n_random=n_random,
            top_k=top_k,
            maxiter=maxiter,
            rng_seed=rng_seed + index
        )

        mle = result['logL_mle']

        # Calculate BIC using the refined maximum likelihood
        k = ar.num_arima_params(
            order,
            include_mean=include_mean,
            include_scale=include_scale
        )

        bic = float(ar.compute_bic(mle, k, n_data))

        result.update({
            'order': order,
            'NSMaxLogL': ns_ref,
            'BIC': bic
        })

        results.append(result)

        print(
            f"NS max: {ns_ref:.5f} | "
            f"MLE: {mle:.5f} | "
            f"Gain: {result['gain']:.5f} | "
            f"BIC: {bic:.5f}"
        )

        

        # Update evidence record
        if fields is not None:

            fields['NSMaxLogL'] = repr(ns_ref)
            fields['MaxLogL'] = repr(float(mle))
            fields['BIC'] = repr(bic)
            fields['MLEGain'] = repr(float(result['gain']))
            fields['MLESuccessfulRuns'] = str(
                result['successful_runs']
            )

            output_records.append(fields)

    # Rewrite evidence file
    if evidence_file is not None:
   
        with open(out_file, 'w') as f:

            for fields in output_records:
                f.write(
                    ', '.join(
                        f"{key}={value}"
                        for key, value in fields.items()
                    ) + '\n'
                )

        print(f"\nUpdated evidence file saved to: {out_file}")

    if results:

        gains = np.array([
            r['gain'] for r in results
        ])

        print("\n========== MLE SUMMARY ==========")
        print(f"Models optimised: {len(results)}")
        print(f"Median gain: {np.median(gains):.6f} nats")
        print(f"Maximum gain: {np.max(gains):.6f} nats")
        print(f"Models improved: {(gains > 1e-3).sum()}")
        print("================================")

    return results