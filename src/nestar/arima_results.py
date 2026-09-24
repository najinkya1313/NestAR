"""
arima_results.py

For postprocessing the output of an ARIMA nested-sampling run.
This includes results of the model-comparison grid (an evidence text file) or a single model's
posterior chain (an anesthetic-format samples CSV). It includes two main classes:

    EvidenceResults  -- read + plot interface for a saved
        evidence-comparison text file (as written by
        ARIMA_model_comparison.run()). Mirrors that class's
        .load_evidence_file() / .plot_evidence_heatmap() / .compare() /
        .prior_volume_report() methods, but standalone: it only needs the
        saved file, not jax / ARIMA_ns / a re-run of the sampler.

    PosteriorResults -- self-consistent read + analysis interface for a
        saved posterior chain (the anesthetic CSV written by
        NestedSamples.to_csv(), as produced by ARIMA_Nested_Sampler /
        ARIMA_model_comparison). Covers both full-chain forecasting
        (compute_forecast/outsample_forecast, insample_forecast -- these
        propagate the whole posterior through fgivenx) and point-estimate
        analysis (posterior_point_estimate, point_estimate_forecast,
        plot_point_estimate_fit, corner_plot).


This module is utilized by arima_model_comparison.py

"""

import ast
import re
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from anesthetic import make_2d_axes, read_chains
from fgivenx import plot_contours, plot_lines
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from .ARIMA import ARIMA_fast, ARIMA_forecast
from .ARIMA_ns import pacf_to_arma


# --------------------------------------------------------------------------- #
# BIC
# --------------------------------------------------------------------------- #

def num_arima_params(order, include_mean=True, include_scale=True):
    """
    Count the number of free parameters in an ARIMA model.

    By default, the parameter count includes the ``p`` AR coefficients
    and ``q`` MA coefficients, along with one parameter for the
    long-term mean ``mu`` and one parameter for the noise scale
    ``sigma``. This matches the parameterisation used by
    :class:`ARIMA_Nested_Sampler`.

    The differencing order ``d`` does not contribute an additional
    free parameter.

    Parameters
    ----------
    order : tuple of int
        ARIMA model order ``(p, d, q)``, where ``p`` is the
        autoregressive order, ``d`` is the differencing order, and
        ``q`` is the moving-average order.
    include_mean : bool, optional
        If ``True``, include the long-term mean ``mu`` in the parameter
        count. Defaults to ``True``.
    include_scale : bool, optional
        If ``True``, include the noise scale ``sigma`` in the parameter
        count. Defaults to ``True``.

    Returns
    -------
    int
        Number of free parameters in the ARIMA model.
    """
    p, d, q = order
    k = 2 * p + q
    if include_mean:
        k += 1
    if include_scale:
        k += 1
    return k


def compute_bic(max_loglikelihood, num_params, num_data):
    """
    Compute the Bayesian Information Criterion (BIC).

    The BIC is defined as

    ``BIC = k * ln(n) - 2 * ln(L_max)``,

    where ``k`` is the number of free parameters, ``n`` is the number
    of data points, and ``L_max`` is the maximum likelihood attained
    by the model.

    Parameters
    ----------
    max_loglikelihood : float
        Maximum log-likelihood attained by the model. 
    num_params : int
        Number of free parameters ``k``. 
    num_data : int
        Number of data points ``n`` used to fit the model.

    Returns
    -------
    float
        Bayesian Information Criterion for the model.
    """
    return num_params * np.log(num_data) - 2 * max_loglikelihood


# --------------------------------------------------------------------------- #
# Prior-volume bookkeeping 
# --------------------------------------------------------------------------- #

def compute_log_V_boost(V):
    """-log(V), the evidence boost from renormalising the rejection-sampled
    prior onto the stationary/invertible region S.

    This will be zero for the PACF priors.
    """
    if V is None:
        return None
    V = np.asarray(V, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(V > 0, -np.log(V), np.inf)


# --------------------------------------------------------------------------- #
# Reading a saved evidence file
# --------------------------------------------------------------------------- #

def _split_top_level(line, sep=","):
    """Split on `sep`, but ignores any `sep` found inside parentheses
    because the Order field is itself a comma-containing tuple like
    '(1, 0, 2)'.
    """
    parts, depth, current = [], 0, []
    for ch in line:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _parse_line(line):
    """Parse a 'Key=value, Key=value, ...' line into a dict of strings."""
    fields = {}
    for part in _split_top_level(line, ","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    return fields


def load_evidence_file(file_name, check_normalization=True, atol=1e-6):
    """
Read back the results written by ``ARIMA_Model_Comparison.run()``.

The function reads an evidence file and extracts whichever fields are
present. This provides backward compatibility with older evidence files
that contain only the ``Order``, ``Seed``, ``Evidence``, and ``Error``
columns. Missing fields are returned as ``None``.

If the file does not contain a ``Posterior`` column, the log posterior
probabilities are recomputed from the ``Evidence`` column. The
``log_V_boost`` quantity is never read directly from the file; it is
always rederived from ``V`` using ``compute_log_V_boost()``.

Parameters
----------
file_name : str
    Path to the evidence file generated by
    ``ARIMA_Model_Comparison.run()``.
check_normalization : bool, optional
    If ``True``, check that the calculated log posterior probabilities
    are normalized. Defaults to ``True``.
atol : float or int, optional
    Tolerance for the deviation of the posterior normalization from 1.
    Defaults to ``1e-6``.

Returns
-------
dict
    Dictionary containing the following entries:

    - ``orders`` : list of tuple or None
        List of ``(p, d, q)`` model orders, or ``None`` if not present.
    - ``evidences`` : np.ndarray
        Log Bayesian evidences for each model.
    - ``evidence_err`` : np.ndarray
        Uncertainties in the log Bayesian evidences.
    - ``log_posteriors`` : np.ndarray
        Normalized log posterior probabilities for each model.
    - ``V`` : np.ndarray or None
        Rejection-sampling acceptance rates.
    - ``max_loglikelihood`` : np.ndarray or None
        Maximum log-likelihood attained by each model.
    - ``BIC`` : np.ndarray or None
        Bayesian Information Criterion value for each model.
    - ``d0_occam`` : np.ndarray or None
        Per-model total D0 Occam-factor contribution.
    - ``log_V_boost`` : np.ndarray or None
        ``-log(V)`` for each model, derived from ``V`` rather than read
        directly from the file.
    - ``net_prior_volume_effect`` : np.ndarray or None
        Sum of ``log_V_boost`` and ``d0_occam``. If a
        ``NetPriorVolume`` column is present, its values are used;
        otherwise the quantity is recomputed when both contributing
        terms are available.

Notes
-----
Per-parameter D0 detail, specifying how much each ``init_y_i`` 
contributed, is not persisted to the text file; only the per-model
total is stored. This detail is available only on a live
``ARIMA_Model_Comparison.d0_per_param`` attribute immediately after
``run()`` and is not available after reloading with
``load_evidence_file()``.
"""
    records = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fields = _parse_line(line)
                if "Evidence" not in fields:
                    raise ValueError("no Evidence field on this line")
                records.append(fields)
            except Exception as e:
                print(f"Skipping line due to parse error: {line}\nError: {e}")

    if not records:
        raise ValueError(f"No parseable records found in {file_name}")

    def _get(key, cast=float):
        if key not in records[0]:
            return None
        return np.array([cast(r[key]) for r in records])

    orders = None
    if "Order" in records[0]:
        orders = [ast.literal_eval(r["Order"]) for r in records]

    evidences = _get("Evidence")
    evidence_err = _get("Error")
    V = _get("V")
    max_loglikelihood = _get("MaxLogL")
    BIC = _get("BIC")
    d0_occam = _get("D0Occam")

    posterior = _get("Posterior")
    if posterior is not None:
        log_posteriors = posterior
    else:
        log_posteriors = evidences - logsumexp(evidences)

    if check_normalization:
        check_posteriors_sum_to_one(log_posteriors, atol=atol)

    log_V_boost = compute_log_V_boost(V)

    net_prior_volume_effect = _get("NetPriorVolume")
    if net_prior_volume_effect is None and log_V_boost is not None and d0_occam is not None:
        net_prior_volume_effect = log_V_boost + d0_occam

    return {
        "orders": orders,
        "evidences": evidences,
        "evidence_err": evidence_err,
        "log_posteriors": log_posteriors,
        "V": V,
        "max_loglikelihood": max_loglikelihood,
        "BIC": BIC,
        "d0_occam": d0_occam,
        "log_V_boost": log_V_boost,
        "net_prior_volume_effect": net_prior_volume_effect,
    }


def check_posteriors_sum_to_one(log_posteriors, atol=1e-6):
    """
    Check whether model posterior probabilities sum to approximately one.

    The function computes the sum of the posterior probabilities from
    their logarithms and issues a warning if the result is not within
    the specified absolute tolerance of one. It never raises an
    exception because of a normalization mismatch.

    Parameters
    ----------
    log_posteriors : array-like
        Log posterior probabilities of the models.
    atol : float, optional
        Absolute tolerance used when checking whether the posterior
        probabilities sum to one. Defaults to ``1e-6``.

    Returns
    -------
    float
        Sum of the model posterior probabilities.
    """
    total = float(np.sum(np.exp(log_posteriors)))
    if not np.isclose(total, 1.0, atol=atol):
        warnings.warn(
            f"Model posterior probabilities sum to {total:.6f}, not 1. "
            "Check the evidence file / normalization (e.g. a model may be "
            "missing, or evidences may contain NaN/-inf).",
            stacklevel=2,
        )
    return total


# --------------------------------------------------------------------------- #
# Plotting (heatmaps)
# --------------------------------------------------------------------------- #

def _grid_from_orders(orders, values, errors, max_p, max_q):
    """Scatter a flat per-model array onto a 2D ``(q, p)`` grid.

    Places each model's value at row ``q`` and column ``p`` of an array initialised
    to NaN, so that models absent from ``orders`` (for example ``(0, d, 0)``) or
    lying outside the requested grid remain NaN.

    Parameters
    ----------
    orders : sequence of tuple of int
        ARIMA orders ``(p, d, q)``, one per entry of ``values``.
    values : array-like
        Flat array of the per-model quantity, ordered like ``orders``.
    errors : array-like or None
        Flat array of uncertainties on ``values``, ordered like ``orders``. If
        None, no error grid is built.
    max_p : int
        Maximum autoregressive order; the grid has ``max_p + 1`` columns.
    max_q : int
        Maximum moving-average order; the grid has ``max_q + 1`` rows.

    Returns
    -------
    heatmap : numpy.ndarray
        Array of shape ``(max_q + 1, max_p + 1)`` holding ``values``, indexed as
        ``heatmap[q, p]``, with NaN in unfilled cells.
    heatmap_err : numpy.ndarray or None
        Array of the same shape holding ``errors``, or None if ``errors`` is None.

    Notes
    -----
    The differencing order ``d`` is ignored when placing values: if several orders
    share the same ``(p, q)``, later entries overwrite earlier ones.
    """
    heatmap = np.full((max_q + 1, max_p + 1), np.nan)

    heatmap_err = (
        np.full((max_q + 1, max_p + 1), np.nan)
        if errors is not None else None
    )

    for i, (p, d, q) in enumerate(orders):
        if p > max_p or q > max_q:
            continue

        heatmap[q, p] = values[i]

        if errors is not None:
            heatmap_err[q, p] = errors[i]

    return heatmap, heatmap_err


def plot_evidence_heatmap(data, max_p, max_q=None, orders=None, contrast=0,
                           highlight_max=False, annotate=True, invert=False,
                           axes=None, figure=None, title=None, cbar_label=None,
                           value_fmt="{:.1f}", cbar_label_top=False, **kwargs):
                               
    """Plot a per-model quantity as an annotated heatmap on the ``ARMA(p, q)`` grid.

    Draws a colour-mapped image of a per-model quantity (for example log
    posterior, BIC, ``log_V``, ``d0_occam`` or ``net_prior_volume_effect``) with the
    AR order ``p`` on the x-axis and the MA order ``q`` on the y-axis. Each cell
    can be annotated with its value and, if available, its uncertainty.

    Parameters
    ----------
    data : tuple of (two array-likes or one array-like and None)
        ``(values, errors)``. ``values`` is either

        - a flat array ordered like ``orders`` (the historical behaviour), in
          which case ``orders`` must also be given, or
        - an already-gridded 2D array of shape ``(max_q + 1, max_p + 1)``.

        ``errors`` may be None if the quantity has no associated uncertainty
        (e.g. BIC, ``V``, ``d0_occam``); the annotations then omit the
        uncertainty line. If given, it must have the same layout as ``values``.
        
    max_p : int
        Maximum autoregressive order of the grid.
    max_q : int, optional
        Maximum moving-average order of the grid. Defaults to ``max_p`` (square
        grid).
    orders : list of tuple of int, optional
        ARIMA orders ``(p, d, q)`` matching a flat ``values``. Required when
        ``values`` is 1D; ignored otherwise.
    contrast : float, default 0
        Offset added to the smallest finite value to set the lower colour limit.
        A positive value raises the lower limit so that low-valued cells
        saturate, making differences among the best-scoring models easier to see.
    highlight_max : bool, default False
        If True, circle the best cell in cyan: the maximum, or the minimum if
        ``invert`` is True.
    annotate : bool, default True
        If True, write each cell's value (and uncertainty, if available) on it.
    invert : bool, default False
        If True, use the reversed colormap so that *lower* values are brighter.
        Use this for quantities where lower is better, e.g. BIC.
    axes : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    figure : matplotlib.figure.Figure, optional
        Figure that owns ``axes``, used to attach the colourbar. Defaults to
        ``axes.figure``. Ignored when ``axes`` is None.
    title : str, optional
        Axes title. No title is drawn if None.
    cbar_label : str, optional
        Colourbar label. Defaults to a LaTeX label for the log posterior.
    value_fmt : str, default "{:.1f}"
        ``str.format`` template for the annotated value. The uncertainty is
        always shown to one decimal place.
    cbar_label_top : bool, default False
        If True, draw ``cbar_label`` as a slightly smaller title above the
        colourbar instead of as a side label. This avoids the side label
        crowding a neighbouring subplot's y-axis in a 1x2 layout and is what
        ``plot_comparison_heatmap`` uses; single-panel calls default to the
        side label.
    **kwargs
        Additional options:

        - ``fig_width``, ``fig_height`` : size in inches of the new figure
          (default: double-column width of 508 pt, about 7.03 in, and 0.6 times
          that for the height). Ignored when ``axes`` is given.
        - ``annotate_fontsize`` : font size of the cell annotations (default 6).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.

    Raises
    ------
    ValueError
        If ``values`` is 1D and ``orders`` is not given.

    Notes
    -----
    The tick range and annotations follow the shape of the gridded array rather
    than ``max_p`` and ``max_q`` if the two differ.
    
    """
    if max_q is None:
        max_q = max_p

    values, errors = data
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        if orders is None:
            raise ValueError("orders= must be given when values is a flat array")
        heatmap_data, heatmap_err = _grid_from_orders(orders, values, errors, max_p, max_q)
    else:
        heatmap_data = values
        heatmap_err = np.asarray(errors, dtype=float) if errors is not None else None

    # Use the actual grid size produced/contained in `heatmap_data` so that
    # plotting ticks/annotations match the array shape even if we expanded
    # the grid in _grid_from_orders.
    max_q = heatmap_data.shape[0] - 1
    max_p = heatmap_data.shape[1] - 1

    finite_vals = heatmap_data[np.isfinite(heatmap_data)]
    vmin, vmax = np.min(finite_vals) + contrast, np.max(finite_vals)

    # --- consistent figure size (double column) ---
    fig_width_pt, inches_per_pt, golden_mean = 508.0, 1.0 / 72.27, 0.6
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width_pt * inches_per_pt * golden_mean
    width = kwargs.get("fig_width", fig_width)
    height = kwargs.get("fig_height", fig_height)
    fontsize = kwargs.get("annotate_fontsize", 6)

    if axes is not None:
        ax = axes
        fig = figure if figure is not None else ax.figure
    else:
        fig, ax = plt.subplots(figsize=(width, height))

    cmap = "inferno_r" if invert else "inferno"
    im = ax.imshow(heatmap_data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    label_text = cbar_label if cbar_label is not None else r"$\log{P_i}$"
    if cbar_label_top:
        cbar.ax.set_title(label_text, fontsize=7, pad=6)
    else:
        cbar.set_label(label_text, fontsize=8)

    ax.set_xlabel("AR (p)", fontsize=9)
    ax.set_ylabel("MA (q)", fontsize=9)
    ax.set_xticks(np.arange(max_p + 1))
    ax.set_yticks(np.arange(max_q + 1))
    ax.set_xticklabels(np.arange(max_p + 1))
    ax.set_yticklabels(np.arange(max_q + 1))
    ax.tick_params(axis="both", labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)

    if highlight_max:
        best = np.nanargmax(heatmap_data) if not invert else np.nanargmin(heatmap_data)
        j, i = np.unravel_index(best, heatmap_data.shape)
        ax.scatter(i, j, s=40, facecolors="none", edgecolors="cyan", linewidths=1)

    if annotate:
        for i in range(max_p + 1):
            for j in range(max_q + 1):
                if not np.isnan(heatmap_data[j, i]):
                    label = value_fmt.format(heatmap_data[j, i])
                    if heatmap_err is not None and not np.isnan(heatmap_err[j, i]):
                        label += f"\n\u00b1{heatmap_err[j, i]:.1f}"
                    ax.text(i, j, label, ha="center", va="center",
                            color="black", fontsize=fontsize, linespacing=1.2)

    fig.tight_layout(pad=0.5)
    return fig


def plot_comparison_heatmap(data1, data2, max_p, max_q=None, orders=None,
                             labels=("Quantity 1", "Quantity 2"),
                             invert=(False, False), highlight_max=(False, False),
                             annotate=True, value_fmt=("{:.1f}", "{:.2f}"),
                             title=None, cbar_label=None, fontsize=(5, 5),
                             contrast=(0, 0),
                             **kwargs):
    """
    Plot two per-model quantities side by side on the same ``(p, q)`` grid.

    Creates a 1x2 figure of annotated heatmaps, for example log posterior versus
    BIC, versus ``V``, or versus ``net_prior_volume_effect``. If one of the two
    datasets is None, a single ordinary heatmap of the other is returned instead,
    using that panel's settings.

    Parameters
    ----------
    data1, data2 : tuple of (array-like, array-like or None), or None
        Data for the left and right panels, each a ``(values, errors)`` tuple
        exactly as for the ``data`` argument of ``plot_evidence_heatmap``.
    max_p : int
        Maximum autoregressive order of the grid.
    max_q : int, optional
        Maximum moving-average order of the grid. Defaults to ``max_p``.
    orders : list of tuple of int, optional
        ARIMA orders ``(p, d, q)`` matching flat ``values`` arrays. Required if
        either dataset is a flat array.
    labels : tuple of str, default ("Quantity 1", "Quantity 2")
        Names of the two quantities, used as panel titles and colourbar labels
        unless ``title`` or ``cbar_label`` are given.
    invert : tuple of bool, default (False, False)
        Per-panel flag to reverse the colormap, for quantities where lower is
        better (e.g. BIC).
    highlight_max : tuple of bool, default (False, False)
        Per-panel flag to circle the best cell.
    annotate : bool, default True
        Whether to annotate cells with their values; applies to both panels.
    value_fmt : tuple of str, default ("{:.1f}", "{:.2f}")
        Per-panel ``str.format`` templates for the annotated values.
    title : tuple of str, optional
        Per-panel titles. Defaults to ``labels``.
    cbar_label : tuple of str, optional
        Per-panel colourbar labels. Defaults to ``labels``.
    fontsize : tuple of int, default (5, 5)
        Per-panel annotation font sizes, forwarded to ``plot_evidence_heatmap``
        as ``annotate_fontsize``.
    contrast : float or tuple of float, default (0, 0)
        Per-panel offsets to the lower colour limit, as in
        ``plot_evidence_heatmap``. A bare scalar ``x`` is shorthand for
        ``(x, x)``.
    **kwargs
        ``fig_width`` and ``fig_height`` set the figure size in inches (default
        width is 1.9 times the single-panel width). All other keyword arguments
        are forwarded to ``plot_evidence_heatmap`` for each panel.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure (or the single-panel figure if one dataset is None).

    Raises
    ------
    ValueError
        If both ``data1`` and ``data2`` are None.
    """
    if data2 is None and data1 is None:
        raise ValueError("At least one of data1, data2 must be given")

    # allow a single scalar (contrast=5) as shorthand for (5, 5)
    if not isinstance(contrast, (tuple, list)):
        contrast = (contrast, contrast)

    if title is None:
        title = labels

    if cbar_label is None:
        cbar_label = labels

    if data2 is None:
        return plot_evidence_heatmap(data1, max_p, max_q=max_q, orders=orders,
                                      invert=invert[0], highlight_max=highlight_max[0],
                                      annotate=annotate, value_fmt=value_fmt[0],
                                      title=title[0], cbar_label=cbar_label[0],
                                      annotate_fontsize=fontsize[0], contrast=contrast[0],
                                      cbar_label_top=True, **kwargs)

    if data1 is None:
        return plot_evidence_heatmap(data2, max_p, max_q=max_q, orders=orders,
                                      invert=invert[1], highlight_max=highlight_max[1],
                                      annotate=annotate, value_fmt=value_fmt[1],
                                      title=title[1], cbar_label=cbar_label[1],
                                      annotate_fontsize=fontsize[1], contrast=contrast[1],
                                      cbar_label_top=True, **kwargs)

    if max_q is None:
        max_q = max_p

    fig_width_pt, inches_per_pt, golden_mean = 508.0, 1.0 / 72.27, 0.6
    fig_width = fig_width_pt * inches_per_pt * 1.9
    fig_height = fig_width_pt * inches_per_pt * golden_mean
    width = kwargs.pop("fig_width", fig_width)
    height = kwargs.pop("fig_height", fig_height)

    fig, axes = plt.subplots(1, 2, figsize=(width, height))

    plot_evidence_heatmap(data1, max_p, max_q=max_q, orders=orders, axes=axes[0],
                           figure=fig, invert=invert[0], highlight_max=highlight_max[0],
                           annotate=annotate, value_fmt=value_fmt[0],
                           title=title[0], cbar_label=cbar_label[0],
                           annotate_fontsize=fontsize[0], contrast=contrast[0],
                           cbar_label_top=True, **kwargs)

    plot_evidence_heatmap(data2, max_p, max_q=max_q, orders=orders, axes=axes[1],
                           figure=fig, invert=invert[1], highlight_max=highlight_max[1],
                           annotate=annotate, value_fmt=value_fmt[1],
                           title=title[1], cbar_label=cbar_label[1],
                           annotate_fontsize=fontsize[1], contrast=contrast[1],
                           cbar_label_top=True, **kwargs)

    fig.tight_layout(pad=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Forecast baselines & metrics
# --------------------------------------------------------------------------- #

def climatology_forecast(train_data, n_forecast):
    """
    Forecast the training-period mean as a flat baseline.

    Parameters
    ----------
    train_data : array-like
        Training-period observations.
    n_forecast : int
        Number of forecast steps.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_forecast,)`` in which every element is the mean of
        ``train_data``.
    """
    return np.full(n_forecast, float(np.mean(np.asarray(train_data))))


def persistence_forecast(train_data, n_forecast):
    """
    Forecast the last training-period value as a flat baseline.

    Parameters
    ----------
    train_data : array-like
        Training-period observations.
    n_forecast : int
        Number of forecast steps.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_forecast,)`` in which every element is the final
        value of ``train_data``.
    """
    return np.full(n_forecast, float(np.asarray(train_data)[-1]))


def forecast_metrics(y_true, y_pred, sigma_pred=None):
    """
    Compute forecast error metrics and, optionally, the log predictive density.

    The root-mean-square error (RMSE) and mean absolute error (MAE) are always
    computed. If ``sigma_pred`` is given, the log predictive density (LPD) of the
    observations under independent Gaussian predictive distributions
    ``N(y_pred, sigma_pred**2)`` is also computed, summed over all forecast steps.

    Parameters
    ----------
    y_true : array-like
        Observed values over the forecast horizon.
    y_pred : array-like
        Predicted values, with the same shape as ``y_true``.
    sigma_pred : float or array-like, optional
        Predictive standard deviation, either a constant or one value per
        forecast step (anything broadcastable to the shape of ``y_true``). If
        None, the LPD is not computed.

    Returns
    -------
    dict of {str: float}
        Dictionary with keys ``'RMSE'`` and ``'MAE'`` and, if ``sigma_pred`` was
        given, ``'LPD'`` 
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    metrics = {'RMSE': float(np.sqrt(np.mean(resid**2))), 'MAE': float(np.mean(np.abs(resid)))}
    if sigma_pred is not None:
        sigma_pred = np.broadcast_to(np.asarray(sigma_pred, dtype=float), y_true.shape)
        metrics['LPD'] = float(np.sum(-0.5*np.log(2*np.pi*sigma_pred**2) - 0.5*resid**2/sigma_pred**2))
    return metrics


def skill_score(metric_model, metric_baseline):
    """Compute the skill of a model relative to a baseline.

    The score is ``1 - metric_model / metric_baseline``. For error metrics where
    lower is better (e.g. RMSE, MAE) it is positive when the model beats the
    baseline, zero when the two are equal, and negative when the model is worse;
    a value of 1 corresponds to a perfect model.

    Parameters
    ----------
    metric_model : float
        Error metric of the model.
    metric_baseline : float
        Error metric of the baseline forecast.

    Returns
    -------
    float
        The skill score, or NaN if ``metric_baseline`` is zero.
    """
    return 1.0 - metric_model/metric_baseline if metric_baseline != 0 else np.nan


# --------------------------------------------------------------------------- #
# EvidenceResults
# --------------------------------------------------------------------------- #

class EvidenceResults:
    """Read-only interface to a saved ARIMA model-comparison evidence file.

    Loads the text file written by ``ARIMA_model_comparison.run()`` and exposes
    the per-model quantities it contains together with plotting and summary
    methods (``plot_evidence_heatmap``, ``compare``, ``prior_volume_report`` and
    ``cumulative_posterior_mass``). It is the sampler-free counterpart to
    ``ARIMA_model_comparison``: it is built straight from the file, has no
    ``jax`` / ``ARIMA_ns`` dependency, and cannot (and need not) run the model
    grid itself.

    Parameters
    ----------
    file_name : str
        Path to the evidence file.
    max_p : int
        Maximum AR order of the model grid.
    max_q : int, optional
        Maximum MA order of the model grid. Defaults to ``max_p``.
    check_normalization : bool, default True
        Passed to ``load_evidence_file``; if True, the loaded model posterior
        probabilities are checked for normalisation to within ``atol``.
    atol : float, default 1e-6
        Absolute tolerance for that normalisation check.

    Attributes
    ----------
    file_name : str
        Path of the file most recently read.
    max_p, max_q : int
        Maximum AR and MA orders of the model grid.
    orders : list of tuple of int
        ARIMA orders ``(p, d, q)`` of the models in the file.
    evidences : array-like
        Per-model evidences.
    evidence_err : array-like
        Uncertainties on the evidences and log posteriors.
    log_posteriors : numpy.ndarray
        Per-model log posterior probabilities.
    V : array-like
        Per-model prior-volume quantity ``V`` stored in the file.
    max_loglikelihood : array-like
        Per-model maximum log-likelihood.
    BIC : array-like
        Per-model Bayesian information criterion.
    d0_occam : array-like
        Per-model D0 Occam penalty.
    log_V_boost : array-like
        Per-model log prior-volume renormalisation from rejection sampling.
    net_prior_volume_effect : array-like
        Net prior-volume contribution to the log posterior (the rejection-sampling
        renormalisation combined with the D0 Occam penalty).

    Each per-model attribute is ordered like ``orders``.
    """

    def __init__(self, file_name, max_p, max_q=None, check_normalization=True, atol=1e-6):
        self.file_name = file_name
        self.max_p = max_p
        self.max_q = max_q if max_q is not None else max_p
        self.orders = None
        self.evidences = None
        self.evidence_err = None
        self.log_posteriors = None
        self.V = None
        self.max_loglikelihood = None
        self.BIC = None
        self.d0_occam = None
        self.log_V_boost = None
        self.net_prior_volume_effect = None
        self.reload(check_normalization=check_normalization, atol=atol)

    def reload(self, file_name=None, check_normalization=True, atol=1e-6):
        """Re-read the evidence file into this instance.

        Useful if the underlying file has been updated since it was loaded, for
        example while a ``run()`` elsewhere is still writing a new line as each model
        finishes.

        Parameters
        ----------
        file_name : str, optional
            New path to read. If given, it replaces ``self.file_name``; if None, the
            current ``self.file_name`` is re-read.
        check_normalization : bool, default True
            Passed to ``load_evidence_file``; see the class docstring.
        atol : float, default 1e-6
            Absolute tolerance for the normalisation check.

        Returns
        -------
        EvidenceResults
            This instance, to allow chaining.
        """
        self.file_name = file_name or self.file_name
        results = load_evidence_file(self.file_name, check_normalization=check_normalization, atol=atol)
        self.orders = results["orders"]
        self.evidences = results["evidences"]
        self.evidence_err = results["evidence_err"]
        self.log_posteriors = results["log_posteriors"]
        self.V = results["V"]
        self.max_loglikelihood = results["max_loglikelihood"]
        self.BIC = results["BIC"]
        self.d0_occam = results["d0_occam"]
        self.log_V_boost = results["log_V_boost"]
        self.net_prior_volume_effect = results["net_prior_volume_effect"]
        return self

    def _quantity(self, name):
        """Look up a stored quantity by name for plotting.

        Parameters
        ----------
        name : str
            One of ``'log_posteriors'``, ``'evidences'``, ``'BIC'``, ``'V'``,
            ``'max_loglikelihood'``, ``'d0_occam'``, ``'log_V_boost'`` or
            ``'net_prior_volume_effect'``.

        Returns
        -------
        values : array-like
            The per-model values.
        errors : array-like or None
            ``evidence_err`` for ``'log_posteriors'`` and ``'evidences'``; None for
            all other quantities.

        Raises
        ------
        ValueError
            If ``name`` is unknown, or if the requested quantity was not populated
            from the file.
        """
        table = {
            "log_posteriors": (self.log_posteriors, self.evidence_err),
            "evidences": (self.evidences, self.evidence_err),
            "BIC": (self.BIC, None),
            "V": (self.V, None),
            "max_loglikelihood": (self.max_loglikelihood, None),
            "d0_occam": (self.d0_occam, None),
            "log_V_boost": (self.log_V_boost, None),
            "net_prior_volume_effect": (self.net_prior_volume_effect, None),
        }
        if name not in table:
            raise ValueError(f"Unknown quantity '{name}'. Choose from {list(table)}.")
        values, errors = table[name]
        if values is None:
            raise ValueError(f"self.{name} is not populated in this evidence file.")
        return values, errors

    def plot_evidence_heatmap(self, quantity="log_posteriors", **kwargs):
        """Plot a single heatmap of one quantity over the ``(p, q)`` grid.

        Parameters
        ----------
        quantity : str or array-like, default "log_posteriors"
            Either the name of a stored quantity (``'log_posteriors'``, ``'BIC'``,
            ``'V'``, ``'evidences'``, ``'max_loglikelihood'``, ``'d0_occam'``,
            ``'log_V_boost'`` or ``'net_prior_volume_effect'``) or a custom array of
            values to plot directly, ordered like ``self.orders`` or already
            gridded. Custom arrays are plotted without error bars.
        **kwargs
            Forwarded to the module-level ``plot_evidence_heatmap``. ``invert``
            defaults to True only for ``'BIC'`` (lower is better). ``cbar_label``
            defaults to ``'log $P_i$'`` whatever quantity is plotted, so override it
            for other quantities.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the heatmap.

        Raises
        ------
        ValueError
            If a named quantity is unknown or not populated in the file.
        """
        if type(quantity) != str:
            data = (quantity, None)  # take custom quantities for plotting on grid.
        else:
            data = self._quantity(quantity)
        if isinstance(quantity, str):
            kwargs.setdefault("invert", quantity == "BIC")
        else:
            kwargs.setdefault("invert", False)  # lower BIC is better

        kwargs.setdefault("cbar_label", f"log $P_i$")
        return plot_evidence_heatmap(data, self.max_p, max_q=self.max_q,
                                      orders=self.orders, **kwargs)

    def compare(self, quantity1="log_posteriors", quantity2=None, labels=None, **kwargs):
        """Compare two quantities side by side on the ``(p, q)`` grid.

        If ``quantity2`` is None this falls back to the ordinary single heatmap of
        ``quantity1`` (same as ``plot_evidence_heatmap``).

        Parameters
        ----------
        quantity1 : str or array-like, default "log_posteriors"
            Quantity for the left panel: a stored quantity name (looked up with
            ``_quantity``) or a raw array of custom values to plot directly, without
            error bars.
        quantity2 : str or array-like, optional
            Quantity for the right panel, in the same forms as ``quantity1``. If
            None, only ``quantity1`` is plotted.
        labels : tuple of str, optional
            Panel titles and colourbar labels. Defaults to the quantity names, or
            "Quantity 1" / "Quantity 2" for custom arrays.
        **kwargs
            Forwarded to ``plot_comparison_heatmap`` (or to ``plot_evidence_heatmap``
            when ``quantity2`` is None). ``invert`` may be given as a per-panel tuple
            and defaults to True for any panel showing ``'BIC'``.

        Returns
        -------
        matplotlib.figure.Figure
            The comparison figure.

        Examples
        --------
        >>> results.compare("log_posteriors", "BIC")
        >>> results.compare("log_posteriors", "V")
        >>> results.compare("log_posteriors", "net_prior_volume_effect")
        """
        if quantity2 is None:
            return self.plot_evidence_heatmap(quantity1, **kwargs)

        data1 = (quantity1, None) if not isinstance(quantity1, str) else self._quantity(quantity1)
        data2 = (quantity2, None) if not isinstance(quantity2, str) else self._quantity(quantity2)

        labels = labels or (
            quantity1 if isinstance(quantity1, str) else "Quantity 1",
            quantity2 if isinstance(quantity2, str) else "Quantity 2",
        )

        invert = kwargs.pop("invert", (quantity1 == "BIC", quantity2 == "BIC"))
        return plot_comparison_heatmap(data1, data2, self.max_p, max_q=self.max_q,
                                        orders=self.orders, labels=labels,
                                        invert=invert, **kwargs)

    def prior_volume_report(self, order_of_interest, baseline_order):
        """Split a log-posterior difference into prior-volume and likelihood parts.

        Quantifies how much of the raw log-posterior difference between two models is
        prior-volume bookkeeping (the rejection-sampling renormalisation combined
        with the D0 Occam penalty, i.e. ``net_prior_volume_effect``) and how much is
        likelihood-driven signal.

        Parameters
        ----------
        order_of_interest : tuple of int
            ARIMA order ``(p, d, q)`` of the model being assessed.
        baseline_order : tuple of int
            ARIMA order ``(p, d, q)`` of the model it is compared against.

        Returns
        -------
        dict
            Dictionary with the following keys. Each "delta" is the value for
            ``order_of_interest`` minus the value for ``baseline_order``.

            - ``'order_of_interest'``, ``'baseline_order'`` : the two orders.
            - ``'raw_log_posterior_delta'`` : delta in ``log_posteriors``.
            - ``'log_V_boost_delta'`` : delta in ``log_V_boost``.
            - ``'d0_occam_delta'`` : delta in ``d0_occam``.
            - ``'net_prior_volume_delta'`` : delta in ``net_prior_volume_effect``.
            - ``'likelihood_only_delta'`` : the raw delta minus the net prior-volume
              delta, i.e. what is left once prior-volume bookkeeping is backed out.
            - ``'fraction_of_raw_delta_from_prior_volume'`` : net prior-volume delta
              divided by the raw delta (NaN if the raw delta is zero).

        Raises
        ------
        ValueError
            If either order is not present in ``self.orders``.

        Examples
        --------
        Sunspot grid:

        >>> results.prior_volume_report(order_of_interest=(9, 0, 1),
        ...                             baseline_order=(0, 0, 1))
        """
        i = self.orders.index(order_of_interest)
        j = self.orders.index(baseline_order)

        raw_delta = float(self.log_posteriors[i] - self.log_posteriors[j])
        prior_vol_delta = float(self.net_prior_volume_effect[i] - self.net_prior_volume_effect[j])
        likelihood_only_delta = raw_delta - prior_vol_delta
        frac_prior_volume = prior_vol_delta / raw_delta if raw_delta != 0 else np.nan

        return {
            "order_of_interest": order_of_interest,
            "baseline_order": baseline_order,
            "raw_log_posterior_delta": raw_delta,
            "log_V_boost_delta": float(self.log_V_boost[i] - self.log_V_boost[j]),
            "d0_occam_delta": float(self.d0_occam[i] - self.d0_occam[j]),
            "net_prior_volume_delta": prior_vol_delta,
            "likelihood_only_delta": likelihood_only_delta,
            "fraction_of_raw_delta_from_prior_volume": frac_prior_volume,
        }

    def cumulative_posterior_mass(self, n):
        """Compute the posterior mass carried by the top-``n`` models.

        Ranks the models by ``log_posteriors`` and reports the cumulative posterior
        probability of the ``n`` best, rather than only the argmax.

        Parameters
        ----------
        n : int
            Number of top-ranked models to include. Clipped to the number of models
            in the grid if larger.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'n'`` : ``n`` after clipping.
            - ``'top_orders'`` : list of the top-``n`` orders ``(p, d, q)``, best
              first, or None if the orders were not loaded from the file.
            - ``'individual_mass'`` : numpy.ndarray with the posterior probability
              of each of those models individually (``exp(log_posteriors)``), best
              first.
            - ``'cumulative_mass'`` : float, the sum of ``individual_mass``, i.e. the
              total posterior probability carried by the top ``n`` models.

        Raises
        ------
        ValueError
            If ``log_posteriors`` is not populated.

        Notes
        -----
        Assumes ``log_posteriors`` are normalised over the grid, so that
        exponentiating them gives probabilities.

        Examples
        --------
        On the sunspot grid the best model carries about 0.17 of the posterior mass
        and the top five just over 0.5:

        >>> results.cumulative_posterior_mass(1)
        >>> results.cumulative_posterior_mass(5)
        """
        if self.log_posteriors is None:
            raise ValueError("self.log_posteriors is not populated in this evidence file.")
        n = min(n, len(self.log_posteriors))
        ranked_idx = np.argsort(self.log_posteriors)[::-1][:n]
        individual_mass = np.exp(self.log_posteriors[ranked_idx])
        top_orders = [self.orders[i] for i in ranked_idx] if self.orders is not None else None
        return {
            "n": n,
            "top_orders": top_orders,
            "individual_mass": individual_mass,
            "cumulative_mass": float(np.sum(individual_mass)),
        }


# --------------------------------------------------------------------------- #
# PosteriorResults -- forecasting helpers
# --------------------------------------------------------------------------- #

def _infer_order_and_prior(chain):
    """Infer the AR/MA orders and prior type from a chain's column names.

    Counts ``phi_N`` / ``theta_N`` columns (normal prior) or ``alpha_ar_N`` /
    ``alpha_ma_N`` columns (PACF prior). If any PACF columns are present the
    PACF prior is assumed.

    Parameters
    ----------
    chain : pandas.DataFrame-like
        Posterior chain (e.g. an ``anesthetic`` samples object) whose column
        labels are either strings or tuples whose first element is the
        parameter name.

    Returns
    -------
    p : int
        Inferred autoregressive order.
    q : int
        Inferred moving-average order.
    prior_type : {'normal', 'pacf'}
        Inferred parametrisation of the ARMA coefficients.

    Notes
    -----
    The differencing order ``d`` is never inferred: it is not a sampled
    parameter, so it leaves no trace in the chain's columns.
    """
    names = [c[0] if isinstance(c, tuple) else c for c in chain.columns]
    names = [str(n) for n in names]
    p_normal = sum(1 for n in names if re.fullmatch(r'phi_\d+', n))
    q_normal = sum(1 for n in names if re.fullmatch(r'theta_\d+', n))
    p_pacf = sum(1 for n in names if re.fullmatch(r'alpha_ar_\d+', n))
    q_pacf = sum(1 for n in names if re.fullmatch(r'alpha_ma_\d+', n))
    if p_pacf or q_pacf:
        return p_pacf, q_pacf, 'pacf'
    return p_normal, q_normal, 'normal'


def _get_arma_coeffs(ar_block, ma_block, prior_type):
    """Convert raw ARMA parameter blocks into AR and MA coefficients.

    For ``prior_type='pacf'`` the blocks are partial-autocorrelation parameters
    and are mapped to coefficients with ``pacf_to_arma``; the MA coefficients are
    negated (``theta = -pacf_to_arma(ma_block)``). For any other prior type
    (``'normal'``) the blocks are already coefficients and are returned as JAX
    arrays.

    Parameters
    ----------
    ar_block : array-like
        Raw autoregressive parameters (coefficients or PACF values).
    ma_block : array-like
        Raw moving-average parameters (coefficients or PACF values).
    prior_type : {'normal', 'pacf'}
        Parametrisation of the blocks.

    Returns
    -------
    phi : jax.Array
        Autoregressive coefficients; empty if ``ar_block`` is empty.
    theta : jax.Array
        Moving-average coefficients; empty if ``ma_block`` is empty.
    """
    if prior_type == 'pacf':
        ar_block, ma_block = jnp.asarray(ar_block), jnp.asarray(ma_block)
        phi = pacf_to_arma(ar_block) if ar_block.shape[0] else jnp.array([])
        theta = -pacf_to_arma(ma_block) if ma_block.shape[0] else jnp.array([])
        return phi, theta
    return jnp.asarray(ar_block), jnp.asarray(ma_block)


def _future_time_index(overall_time, upper_index, num_forecast):
    """Return the time stamps of the forecast horizon.

    If the requested future stamps exist in ``overall_time`` they are read from
    it. Otherwise they are extrapolated forward from its last stamp at its
    (assumed constant) step, which covers the no-test-data case where
    ``overall_time`` spans only the training period.

    Parameters
    ----------
    overall_time : array-like
        Time stamps of the available series.
    upper_index : int
        Index in ``overall_time`` of the first forecast time.
    num_forecast : int
        Number of forecast steps.

    Returns
    -------
    numpy.ndarray
        ``overall_time[upper_index:upper_index + num_forecast]`` if that slice is
        fully available; otherwise ``num_forecast`` extrapolated stamps starting
        one step after ``overall_time[-1]``.

    Notes
    -----
    The step is ``overall_time[1] - overall_time[0]``, or 1 if ``overall_time``
    has a single entry. When extrapolating, the new stamps always start after the
    last entry of ``overall_time``, regardless of ``upper_index``.
    """
    overall_time = np.asarray(overall_time)
    if upper_index + num_forecast <= len(overall_time):
        return overall_time[upper_index:upper_index + num_forecast]
    step = overall_time[1] - overall_time[0] if len(overall_time) > 1 else 1
    start = overall_time[-1] + step
    return start + step * np.arange(num_forecast)


def _weighted_mode(values, weights, grid_size=2000):
    """Estimate the mode of a weighted 1D sample with a Gaussian KDE.

    Fits a Gaussian kernel density estimate to ``values`` with per-sample
    ``weights`` and returns the location of its peak on a regular grid spanning
    the sample range. This is a marginal (per-parameter) mode, not a joint/MAP
    estimate.

    Parameters
    ----------
    values : array-like
        One-dimensional samples.
    weights : array-like
        Per-sample weights, same length as ``values``.
    grid_size : int, default 2000
        Number of grid points on which the density is evaluated.

    Returns
    -------
    float
        Location of the density peak, or the weighted mean of ``values`` if a
        KDE cannot be built (e.g. zero-variance samples).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    try:
        kde = gaussian_kde(values, weights=weights)
    except Exception:
        return float(np.average(values, weights=weights))
    grid = np.linspace(values.min(), values.max(), grid_size)
    density = kde(grid)
    return float(grid[np.argmax(density)])


# --------------------------------------------------------------------------- #
# PosteriorResults
# --------------------------------------------------------------------------- #

class PosteriorResults:
    """Read and analyse a saved ARIMA posterior chain.

    Wraps an ``anesthetic``-format chain, either loaded from the CSV written by
    ``NestedSamples.to_csv()`` (as produced by ``ARIMA_Nested_Sampler`` /
    ``ARIMA_model_comparison``) or supplied as an already-loaded chain such as a
    live ``NestedSamples`` instance. The orders ``p`` and ``q`` and the prior type
    are inferred from the chain's columns if not given; the differencing order
    ``d`` cannot be inferred and defaults to 0.

    Two families of methods are provided:

    - Full-chain forecasting (unchanged from the former ``ARIMAForecaster``):
      ``compute_forecast`` and ``outsample_forecast`` propagate the whole
      posterior through ``fgivenx`` to give out-of-sample forecasts with
      posterior uncertainty bands; ``insample_forecast`` does the same for the
      in-sample fit and residuals, and ``pooled_residuals`` returns the residuals
      as an array.
    - Point-estimate analysis: ``posterior_point_estimate`` collapses the chain
      to a single mean, median or mode per parameter; ``point_estimate_forecast``
      uses that point to compute an in-sample fit (and optionally a forecast)
      directly via ``ARIMA_fast`` / ``ARIMA_forecast``, with no posterior
      propagation; ``plot_point_estimate_fit`` plots that fit against the data
      with residuals; ``corner_plot`` draws the 2D posterior, optionally with the
      prior overlaid.

    Parameters
    ----------
    chain_path : str or anesthetic.Samples
        Path to a chain CSV, or an already-loaded chain object.
    train_data : array-like
        One-dimensional training series to which the chain was fitted.
    order : tuple of int, optional
        ARIMA order ``(p, d, q)``. If None, ``p`` and ``q`` are inferred from the
        chain and ``d`` is taken from the ``d`` argument.
    prior_type : {'normal', 'pacf'}, optional
        Parametrisation of the ARMA coefficients in the chain: ``'normal'``
        (``phi_i`` / ``theta_j`` columns) or ``'pacf'`` (``alpha_ar_i`` /
        ``alpha_ma_j`` columns). Inferred from the chain if None.
    d : int, default 0
        Differencing order used when ``order`` is None; ignored otherwise.
    seed : int, default 0
        Default random seed for generating forecast innovations.

    Attributes
    ----------
    chain : anesthetic.Samples
        The posterior chain.
    train_data : numpy.ndarray
        Training series.
    order : tuple of int
        ARIMA order ``(p, d, q)``.
    prior_type : {'normal', 'pacf'}
        Parametrisation of the ARMA coefficients in the chain.
    seed : int
        Default random seed.
    sigma, residuals
        Set by ``point_estimate_forecast``: the innovation scale used for the
        in-sample fit and the resulting residuals.

    Notes
    -----
    The resolved order and prior type are printed on construction.
    """

    def __init__(self, chain_path, train_data, order=None, prior_type=None, d=0, seed=0):
        self.chain = read_chains(chain_path) if isinstance(chain_path, str) else chain_path
        inferred_p, inferred_q, inferred_prior_type = _infer_order_and_prior(self.chain)
        source = "given" if order is not None else "inferred from chain"
        self.prior_type = prior_type if prior_type is not None else inferred_prior_type
        self.order = order if order is not None else (inferred_p, d, inferred_q)
        self.train_data = np.asarray(train_data)
        self.seed = seed
        print(f"PosteriorResults: order={self.order}, prior_type='{self.prior_type}' ({source})")

    # ------------------------------------------------------------------ #
    # shared key bookkeeping
    # ------------------------------------------------------------------ #

    def _arma_keys(self):
        """Return the chain column names of the ARMA parameters.

        Returns
        -------
        ar_keys : list of str
            ``phi_1`` ... ``phi_p`` for the ``'normal'`` prior, or ``alpha_ar_1`` ...
            ``alpha_ar_p`` for ``'pacf'``.
        ma_keys : list of str
            ``theta_1`` ... ``theta_q`` for the ``'normal'`` prior, or
            ``alpha_ma_1`` ... ``alpha_ma_q`` for ``'pacf'``.
        init_y_keys : list of str
            ``init_y_1`` ... ``init_y_p``.
        """
        p, d, q = self.order
        ar_keys = [f'alpha_ar_{i+1}' for i in range(p)] if self.prior_type == 'pacf' else [f'phi_{i+1}' for i in range(p)]
        ma_keys = [f'alpha_ma_{j+1}' for j in range(q)] if self.prior_type == 'pacf' else [f'theta_{j+1}' for j in range(q)]
        init_y_keys = [f'init_y_{i+1}' for i in range(p)]
        return ar_keys, ma_keys, init_y_keys

    def _param_keys(self):
        """Return the full list of sampled-parameter column names.

        Returns
        -------
        list of str
            AR keys, MA keys, ``'sigma'``, ``'mu'`` and the ``init_y`` keys, in that
            order.
        """
        ar_keys, ma_keys, init_y_keys = self._arma_keys()
        return ar_keys + ma_keys + ['sigma', 'mu'] + init_y_keys

    # ------------------------------------------------------------------ #
    # full-chain (posterior-propagated) forecasting
    # ------------------------------------------------------------------ #

    def _pack_samples(self, n_samples, seed=None):
        """Draw posterior samples and pack them as flat parameter tuples.

        Draws ``n_samples`` rows from the chain and converts each into a tuple laid
        out as ``(ar..., ma..., sigma, mu, init_y..., forecast_seed)``, the format
        consumed by ``_unpack_params`` and the ``fgivenx`` wrapper functions.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw from the chain.
        seed : int, optional
            Seed for the random per-sample forecast seeds (the last element of each
            tuple). Defaults to ``self.seed``.

        Returns
        -------
        list of tuple
            One packed tuple per drawn sample.

        Raises
        ------
        ValueError
            If the chain lacks any column expected for this object's order and prior
            type.

        Notes
        -----
        ``seed`` only controls the per-sample forecast seeds; the selection of rows
        by ``self.chain.sample`` is not seeded here.
        """
        p, d, q = self.order
        ar_keys, ma_keys, init_y_keys = self._arma_keys()

        samples_df = self.chain.sample(n_samples)
        missing = [k for k in ar_keys + ma_keys + ['sigma', 'mu'] + init_y_keys if k not in samples_df.columns]
        if missing:
            raise ValueError(f"chain is missing expected columns for order={self.order}, "
                              f"prior_type='{self.prior_type}': {missing}")

        rng = np.random.RandomState(seed if seed is not None else self.seed)
        forecast_seeds = rng.randint(0, 1_000_000, size=len(samples_df))

        ar_s = [samples_df[k] for k in ar_keys]
        ma_s = [samples_df[k] for k in ma_keys]
        iy_s = [samples_df[k] for k in init_y_keys]

        packed = []
        for i in range(len(samples_df)):
            ar = [ar_s[j].iloc[i] for j in range(p)]
            ma = [ma_s[j].iloc[i] for j in range(q)]
            iy = [iy_s[j].iloc[i] for j in range(p)]
            packed.append(tuple(ar + ma + [float(samples_df['sigma'].iloc[i]), float(samples_df['mu'].iloc[i])]
                                 + iy + [float(forecast_seeds[i])]))
        return packed

    def _unpack_params(self, params):
        """Unpack a packed sample tuple into model quantities.

        Parameters
        ----------
        params : sequence of float
            Flat sample laid out as ``(ar..., ma..., sigma, mu, init_y...,
            forecast_seed)``, as produced by ``_pack_samples``.

        Returns
        -------
        phi : jax.Array
            Autoregressive coefficients (PACF values already converted if the prior
            type is ``'pacf'``).
        theta : jax.Array
            Moving-average coefficients (likewise converted).
        sigma : jax.Array
            Innovation standard deviation (scalar).
        mu : jax.Array
            Process mean (scalar).
        init_y : jax.Array
            Initial values of the ARIMA recurrence.
        seed_i : int
            Random seed for this sample's forecast innovations.
        """
        p, d, q = self.order
        params = jnp.asarray(params)
        ar_raw, ma_raw = params[0:p], params[p:p+q]
        sigma, mu = params[p+q], params[p+q+1]
        init_y = params[p+q+2:p+q+2+p]
        seed_i = int(params[-1])
        phi, theta = _get_arma_coeffs(ar_raw, ma_raw, self.prior_type)
        return phi, theta, sigma, mu, init_y, seed_i

    def _forecast_func(self, num_forecast):
        """Build a forecast function in the form expected by ``fgivenx``.

        Parameters
        ----------
        num_forecast : int
            Number of forecast steps.

        Returns
        -------
        callable
            Function ``f(x, params)`` that unpacks a packed sample ``params`` and
            returns the ``num_forecast``-step forecast from ``ARIMA_forecast``. The
            argument ``x`` only satisfies ``fgivenx``'s calling convention and is
            unused.
        """
        def f(x, params):
            phi, theta, sigma, mu, init_y, seed_i = self._unpack_params(params)
            return ARIMA_forecast(self.train_data, self.order, sigma, mu, phi, theta,
                                   num_forecast, init_y, seed_i)
        return f

    def _fit_func(self,deterministic=True):
        """Build an in-sample fit function in the form expected by ``fgivenx``.

        Parameters
        ----------
        deterministic : bool, default True
            If True, the fit is evaluated with innovation scale ``sigma = 0``
            (noise-free). If False, the sample's own ``sigma`` is passed to
            ``ARIMA_fast``.

        Returns
        -------
        callable
            Function ``f(x, params)`` that unpacks a packed sample ``params`` and
            returns the fitted values over the training period from ``ARIMA_fast``.
            The argument ``x`` only satisfies ``fgivenx``'s calling convention and is
            unused.
        """
        def f(x, params):
            phi, theta, sigma, mu, init_y, seed_i = self._unpack_params(params)
            if deterministic==True:
             return ARIMA_fast(self.train_data, self.order, 0.0, mu, phi, theta, init_y, seed_i)
            else:
             return ARIMA_fast(self.train_data, self.order,sigma, mu, phi, theta, init_y, seed_i)

        return f

    def _residual_func(self,deterministic=True):
        """Build a residual function in the form expected by ``fgivenx``.

        Parameters
        ----------
        deterministic : bool, default True
            Passed to ``_fit_func``: whether the fit is evaluated noise-free.

        Returns
        -------
        callable
            Function ``f(x, params)`` returning ``train_data`` minus the fitted
            values for a packed sample ``params``. The argument ``x`` only satisfies
            ``fgivenx``'s calling convention and is unused.
        """
        fit_func = self._fit_func(deterministic=deterministic)
        def f(x, params):
            return jnp.asarray(self.train_data) - fit_func(x, params)
        return f

    def compute_forecast(self, num_forecast, n_samples=1000, seed=0):
        """Propagate posterior samples through the forecaster.

        Draws ``n_samples`` samples from the chain and replays each through
        ``ARIMA_forecast``, returning the forecast matrix and summary statistics.

        Parameters
        ----------
        num_forecast : int
            Number of forecast steps.
        n_samples : int, default 1000
            Number of posterior samples to draw.
        seed : int, default 0
            Seed for the per-sample forecast seeds; see ``_pack_samples``.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'forecast_matrix'`` : numpy.ndarray of shape
              ``(n_samples, num_forecast)``, one forecast per sample.
            - ``'mean_forecast'`` : numpy.ndarray, mean over samples at each step.
            - ``'sigma_forecast'`` : numpy.ndarray, standard deviation over samples
              at each step.
            - ``'packed_samples'`` : list of the packed sample tuples used.
            - ``'forecast_func'`` : the ``fgivenx``-style forecast function used.
        """
        packed_samples = self._pack_samples(n_samples, seed=seed)
        forecast_func = self._forecast_func(num_forecast)
        forecast_matrix = np.array([
            np.asarray(forecast_func(None, p)) for p in tqdm.tqdm(packed_samples, desc="NS forecast samples")
        ])
        return {
            'forecast_matrix': forecast_matrix,
            'mean_forecast': forecast_matrix.mean(axis=0),
            'sigma_forecast': forecast_matrix.std(axis=0),
            'packed_samples': packed_samples,
            'forecast_func': forecast_func,
        }

    def outsample_forecast(self, overall_time, overall_data, num_forecast, upper_index=None,
                        n_samples=1000, seed=0,
                        plot_nested=True, plot_climatology=True, plot_persistence=True,
                        ax=None,
                        label_fontsize=9, tick_labelsize=7, legend_fontsize=7,
                        title_fontsize=9, show_legend=True, show_title=False,
                        cbar_labelsize=7, cbar_tick_position='right', show_colorbar=True,
                        **kwargs):
     """Plot and score an out-of-sample forecast against simple baselines.

     Propagates the posterior through ``compute_forecast`` and draws the resulting
     forecast density (``fgivenx`` contours) over the forecast horizon, together
     with climatology and persistence baselines and, if test data are available,
     the observations. With test data, RMSE, MAE and LPD are computed for each
     forecast, together with the RMSE-based skill of the posterior-mean forecast
     relative to each baseline, and a summary is printed.

     Parameters
     ----------
     overall_time : array-like
         Time stamps of the full series (training plus any test period). If they
         do not extend to the forecast horizon, the future stamps are
         extrapolated at constant step.
     overall_data : array-like
         Full observed series (training plus test). Only used for comparison with
         the forecast if ``upper_index`` is given.
     num_forecast : int
         Number of forecast steps.
     upper_index : int, optional
         Index in ``overall_data`` at which the forecast period starts, i.e. the
         size of the training set. If None, no test data are assumed: the
         forecast starts at ``len(overall_data)``, no observations are plotted,
         and no metrics or skill scores are computed.
     n_samples : int, default 1000
         Number of posterior samples to propagate.
     seed : int, default 0
         Seed for the per-sample forecast seeds.
     plot_nested : bool, default True
         Whether to compute and plot the posterior-propagated forecast.
     plot_climatology : bool, default True
         Whether to compute and plot the climatology baseline, shaded by plus or
         minus one standard deviation of the training data.
     plot_persistence : bool, default True
         Whether to compute and plot the persistence baseline, shaded likewise.
     ax : matplotlib.axes.Axes, optional
         Axes to draw on. If None, a new figure and axes are created.
     label_fontsize : int, default 9
         Font size of the axis labels.
     tick_labelsize : int, default 7
         Font size of the tick labels.
     legend_fontsize : int, default 7
         Font size of the legend text.
     title_fontsize : int, default 9
         Font size of the title.
     show_legend : bool, default True
         If False, omit the legend entirely.
     show_title : bool, default False
         If True, draw a title (see ``title`` under ``**kwargs``).
     cbar_labelsize : int, default 7
         Tick label size of the ``fgivenx`` colourbar.
     cbar_tick_position : str, default 'right'
         Side of the colourbar on which ticks are drawn, passed to the colourbar
         axis' ``set_ticks_position``.
     show_colorbar : bool, default True
         If True, add a colourbar with the 1, 2 and 3 sigma contour levels.
     **kwargs
         Additional options:

         - ``figsize`` : figure size when a new figure is created (default
           ``(9, 6)``).
         - ``xlabel``, ``ylabel`` : axis labels (defaults "Time" and "Value").
         - ``title`` : title text, used if ``show_title`` is True (default
           ``"ARIMA{order} forecast"``).
         - ``ylim`` : ``(low, high)`` y-axis limits.

     Returns
     -------
     fig : matplotlib.figure.Figure
         The figure containing the forecast plot.
     results : dict
         Results, with keys ``'nested'``, ``'climatology'`` and ``'persistence'``
         (the latter two only if plotted).

         - ``'nested'`` holds the output of ``compute_forecast``.
         - ``'climatology'`` and ``'persistence'`` each hold ``'forecast'``, the
           baseline forecast array.
         - With test data, each also holds ``'metrics'`` (as returned by
           ``forecast_metrics``), and ``'nested'`` additionally holds
           ``'skill_vs_climatology'`` and ``'skill_vs_persistence'`` for each
           baseline that was plotted.

     Notes
     -----
     The predictive spread used for the LPD is the per-step standard deviation over
     posterior samples for the posterior forecast, and the standard deviation of
     the training data for the baselines.
     """
     have_test_data = upper_index is not None
     if upper_index is None:
        upper_index = len(overall_data)

     y_true = np.asarray(overall_data[upper_index:upper_index+num_forecast]) if have_test_data else None
     x_forecast = _future_time_index(overall_time, upper_index, num_forecast)
     train_arr = self.train_data
     results = {}

     fig, ax = (plt.subplots(figsize=kwargs.get("figsize", (9, 6))) if ax is None else (ax.figure, ax))

     if plot_nested:
        ns = self.compute_forecast(num_forecast, n_samples=n_samples, seed=seed)
        cbar = plot_contours(f=ns['forecast_func'], x=x_forecast, samples=ns['packed_samples'], ax=ax)
     if show_colorbar==True:
        cb = plt.colorbar(cbar, ticks=[0, 1, 2, 3], ax=ax)
        cb.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])
        cb.ax.tick_params(labelsize=cbar_labelsize, direction="in")
        cb.outline.set_linewidth(0.5)
        cb.ax.yaxis.set_ticks_position(cbar_tick_position)
     results['nested'] = ns
     if have_test_data:
        results['nested']['metrics'] = forecast_metrics(y_true, ns['mean_forecast'], sigma_pred=ns['sigma_forecast'])

     clim_sigma = float(np.std(train_arr - np.mean(train_arr)))

     if plot_climatology:
        y_clim = climatology_forecast(train_arr, num_forecast)
        results['climatology'] = {'forecast': y_clim}
        if have_test_data:
            results['climatology']['metrics'] = forecast_metrics(y_true, y_clim, sigma_pred=clim_sigma)
        ax.plot(x_forecast, y_clim, '--', color='tab:blue', label='Climatology')
        ax.fill_between(x_forecast, y_clim - clim_sigma, y_clim + clim_sigma, color='tab:blue', alpha=0.15)

     if plot_persistence:
        y_pers = persistence_forecast(train_arr, num_forecast)
        results['persistence'] = {'forecast': y_pers}
        if have_test_data:
            results['persistence']['metrics'] = forecast_metrics(y_true, y_pers, sigma_pred=clim_sigma)
        ax.plot(x_forecast, y_pers, ':', color='tab:green', label='Persistence')
        ax.fill_between(x_forecast, y_pers - clim_sigma, y_pers + clim_sigma, color='tab:green', alpha=0.15)

     if have_test_data and plot_nested:
        if plot_climatology:
            results['nested']['skill_vs_climatology'] = skill_score(
                results['nested']['metrics']['RMSE'], results['climatology']['metrics']['RMSE'])
        if plot_persistence:
            results['nested']['skill_vs_persistence'] = skill_score(
                results['nested']['metrics']['RMSE'], results['persistence']['metrics']['RMSE'])

     if have_test_data:
        ax.plot(x_forecast, y_true, marker='+', lw=0, color='black', label='Observed')

     ax.set_xlabel(kwargs.get("xlabel", "Time"), fontsize=label_fontsize)
     ax.set_ylabel(kwargs.get("ylabel", "Value"), fontsize=label_fontsize)
     ax.tick_params(axis='both', labelsize=tick_labelsize)
     if show_title:
        ax.set_title(kwargs.get("title", f"ARIMA{self.order} forecast"), fontsize=title_fontsize)
     if show_legend:
        ax.legend(fontsize=legend_fontsize)
     if "ylim" in kwargs:
        ax.set_ylim(*kwargs["ylim"])
     fig.tight_layout()

     print(f"--- Forecast summary: ARIMA{self.order} ({self.prior_type} prior), h={num_forecast} ---")
     for name, block in results.items():
        if 'metrics' not in block:
            continue
        m = block['metrics']
        extra = ""
        if name == 'nested':
            if 'skill_vs_climatology' in block: extra += f"  skill_vs_clim={block['skill_vs_climatology']:.3f}"
            if 'skill_vs_persistence' in block: extra += f"  skill_vs_pers={block['skill_vs_persistence']:.3f}"
        print(f"{name:12s}: RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  LPD={m['LPD']:.2f}{extra}")

     return fig, results

    def insample_forecast(self, training_time, n_samples=1000, seed=0, meas_sigma=None,ax=None,deterministic=False, **kwargs):
        """Plot the in-sample posterior fit and residuals against the training data.

        Draws ``n_samples`` samples from the chain and plots their fit lines (top
        panel) and residual lines (bottom panel) with ``fgivenx.plot_lines``, with the
        training data overlaid on the top panel.

        Parameters
        ----------
        training_time : array-like
            Time stamps of the training data.
        n_samples : int, default 1000
            Number of posterior samples to draw.
        seed : int, default 0
            Seed for the per-sample forecast seeds; see ``_pack_samples``.
        meas_sigma : array-like or float, optional
            Measurement uncertainties, drawn as error bars on the data. No error bars
            if None.
        ax : sequence of matplotlib.axes.Axes, optional
            The two axes for the fit and residual panels. If None, a new figure with
            two vertically stacked panels sharing the x-axis is created.
        deterministic : bool, default False
            If True, the fit lines are computed noise-free (``sigma = 0``). Applies
            to the fit panel only: the residual panel is always computed from the
            noise-free fit.
        **kwargs
            Styling options:

            - ``figsize`` : figure size for a new figure (default ``(9, 8)``).
            - ``fit_color`` : colour of the fit lines (default "red").
            - ``lw`` : line width of the fit lines (default 1).
            - ``fmt``, ``alpha``, ``capsize``, ``ms`` : marker format, opacity, error
              bar cap size and marker size of the data points (defaults "o", 1, 2
              and 1).
            - ``data_label`` : legend label of the data (default a LaTeX label for
              D_t).
            - ``xlabel``, ``ylabel`` : axis labels (defaults "Time" and "Value").

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing both panels.
        info : dict
            Dictionary with keys ``'packed_samples'``, ``'fit_func'`` and
            ``'residual_func'``.
        """
        packed_samples = self._pack_samples(n_samples, seed=seed)
        fit_func, residual_func = self._fit_func(deterministic=deterministic), self._residual_func(deterministic=deterministic)

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=kwargs.get("figsize", (9, 8)), sharex=True,
                                    gridspec_kw={'hspace': 0.1})
        else:
            fig = ax[0].figure

        
        
        

        plot_lines(fit_func, training_time, packed_samples, ax=ax[0], color=kwargs.get("fit_color", "red"),linewidth=kwargs.get("lw",1))
        ax[0].errorbar(training_time, self.train_data,yerr=meas_sigma, fmt=kwargs.get("fmt","o"),alpha=kwargs.get("alpha",1),capsize=kwargs.get("capsize",2), ms=kwargs.get("ms",1), lw=0, color='black',
                   label=kwargs.get("data_label", r'$D_t$'))
        ax[0].set_ylabel(kwargs.get("ylabel", "Value"))
        ax[0].plot([], [], c='red', lw=2, label=r'$\hat{y}_t$')
        ax[0].grid(alpha=0.5)
        ax[0].tick_params(axis='both', labelsize=7)

        plot_lines(residual_func, training_time, packed_samples, ax=ax[1])
        ax[1].set_xlabel(kwargs.get("xlabel", "Time"))
        ax[1].set_ylabel(r'Residuals ($D_t - \hat{y}_t$)')
        ax[1].grid(alpha=0.5)
        ax[1].tick_params(axis='both',labelsize=7)

        leg = fig.legend(loc='upper center',bbox_to_anchor=(0.5,0.55),frameon=True,fontsize=7)
        frame = leg.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_alpha(1.0)

        fig.tight_layout(pad=0.3)
        return fig, {'packed_samples': packed_samples, 'fit_func': fit_func, 'residual_func': residual_func}

    # ------------------------------------------------------------------ #
    # point-estimate summaries of the posterior
    # ------------------------------------------------------------------ #

    def posterior_point_estimate(self, estimate='mean', param_keys=None):
        """Collapse the chain to a single point estimate per parameter.

        Parameters
        ----------
        estimate : {'mean', 'median', 'mode'}, default 'mean'
            Type of point estimate. ``'mean'`` and ``'median'`` use the chain's own
            weight-aware ``.mean()`` and ``.median()``. ``'mode'`` is a
            per-parameter (marginal) weighted-KDE mode, not a joint/MAP estimate,
            computed with ``_weighted_mode``.
        param_keys : list of str, optional
            Chain columns to summarise. Defaults to this object's full parameter set
            (ARMA coefficients or PACF alphas, ``sigma``, ``mu`` and ``init_y``).

        Returns
        -------
        dict of {str: float}
            Mapping from column name to point-estimate value.

        Raises
        ------
        ValueError
            If ``estimate`` is not one of ``'mean'``, ``'median'`` or ``'mode'``.
        """
        if estimate not in ('mean', 'median', 'mode'):
            raise ValueError("estimate must be one of 'mean', 'median', 'mode'")
        if param_keys is None:
            param_keys = self._param_keys()

        point = {}
        if estimate in ('mean', 'median'):
            for k in param_keys:
                point[k] = float(getattr(self.chain[k], estimate)())
        else:
            weights = self.chain.get_weights()
            for k in param_keys:
                point[k] = _weighted_mode(self.chain[k].to_numpy(), weights)
        return point
    
    def pooled_residuals(self, n_samples=1000, seed=0, packed_samples=None,deterministic=False):
     """Compute the residuals ``D_t - y_hat_t`` across posterior samples.

     Returns the array behind the residual fan drawn by ``insample_forecast``,
     which only passes the residual function to ``fgivenx.plot_lines`` and never
     keeps the values. No time axis is needed: the ``x`` argument of the residual
     function exists only to satisfy ``fgivenx``'s calling convention and is not
     used in the computation.

     Parameters
     ----------
     n_samples : int, default 1000
         Number of posterior samples to draw. Ignored if ``packed_samples`` is
         given.
     seed : int, default 0
         Seed for the per-sample forecast seeds; see ``_pack_samples``.
     packed_samples : list of tuple, optional
         Pre-packed samples to reuse (e.g. the ``'packed_samples'`` entry returned
         by ``compute_forecast`` or ``insample_forecast``) instead of drawing new
         ones.
     deterministic : bool, default False
         If True, residuals are taken against the noise-free fit (``sigma = 0``).
         Use True to reproduce the residual panel of ``insample_forecast``, which
         always uses the noise-free fit.

     Returns
     -------
     numpy.ndarray
         Array of shape ``(n_samples, n_train)`` of residuals, one row per
         sample.
     """
     packed_samples = packed_samples if packed_samples is not None else self._pack_samples(n_samples, seed=seed)
     residual_func = self._residual_func(deterministic=deterministic)
     residual_matrix = np.array([
        np.asarray(residual_func(None, p)) for p in tqdm.tqdm(packed_samples, desc="Pooled residuals")
    ])
     return residual_matrix

    def point_estimate_forecast(self, estimate='mean', num_forecast=0, deterministic=False,point_params=None, seed=None):
        """Compute an in-sample fit (and optional forecast) from one point estimate.

        Point-estimate analogue of ``compute_forecast`` / ``outsample_forecast``:
        rather than propagating the full chain, the posterior is collapsed to a
        single value per parameter and the in-sample fit is built from it with
        ``ARIMA_fast``. If ``num_forecast > 0``, a forward forecast is also generated
        with ``ARIMA_forecast``. This generalises building a fit directly from the
        posterior means to any of the mean, median or mode.

        Parameters
        ----------
        estimate : {'mean', 'median', 'mode'}, default 'mean'
            Type of point estimate; see ``posterior_point_estimate``. Ignored if
            ``point_params`` is given.
        num_forecast : int, default 0
            Number of steps to forecast beyond the training data. If 0, only the
            in-sample fit is computed.
        deterministic : bool, default False
            If True, the in-sample fit is computed noise-free (``sigma = 0``). This
            does not affect the forecast, which always uses the point-estimate
            ``sigma`` and ``seed`` to generate its innovations.
        point_params : dict, optional
            Precomputed point estimate (e.g. from ``posterior_point_estimate``) to
            reuse instead of recomputing it.
        seed : int, optional
            Random seed passed to ``ARIMA_fast`` and ``ARIMA_forecast``. Defaults to
            ``self.seed``.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'point_params'`` : the point-estimate dict used, plus the resolved
              ``'phi'`` and ``'theta'`` arrays (PACF alphas already transformed).
            - ``'y_fit'`` : numpy.ndarray of in-sample fitted values, the same length
              as ``train_data``.
            - ``'residuals'`` : numpy.ndarray, ``train_data`` minus ``y_fit``.
            - ``'y_forecast'`` : numpy.ndarray of the ``num_forecast``-step forecast,
              or None if ``num_forecast`` is 0.

        Notes
        -----
        As side effects, ``self.sigma`` (0 if ``deterministic``, else the
        point-estimate ``sigma``) and ``self.residuals`` are set.
        """
        point_params = point_params if point_params is not None else self.posterior_point_estimate(estimate=estimate)
        ar_keys, ma_keys, init_y_keys = self._arma_keys()

        ar_raw = jnp.array([point_params[k] for k in ar_keys]) if ar_keys else jnp.array([])
        ma_raw = jnp.array([point_params[k] for k in ma_keys]) if ma_keys else jnp.array([])
        phi, theta = _get_arma_coeffs(ar_raw, ma_raw, self.prior_type)
        sigma, mu = point_params['sigma'], point_params['mu']
        init_y = jnp.array([point_params[k] for k in init_y_keys]) if init_y_keys else jnp.array([])
        seed_i = seed if seed is not None else self.seed
        if deterministic==True:
         self.sigma = 0.0

        else:
         self.sigma= sigma
        
        y_fit = np.asarray(ARIMA_fast(data=self.train_data, order=self.order, sigma=self.sigma, mu=mu, phi=phi, theta=theta, init_y=init_y, seed=seed_i))
        residuals = self.train_data - y_fit
        self.residuals = residuals

        y_forecast = None
        if num_forecast > 0:
            y_forecast = np.asarray(ARIMA_forecast(self.train_data, self.order, sigma, mu, phi, theta,
                                                    num_forecast, init_y, seed_i))

        return {
            'point_params': {**point_params, 'phi': np.asarray(phi), 'theta': np.asarray(theta)},
            'y_fit': y_fit,
            'residuals': residuals,
            'y_forecast': y_forecast,
        }

    def plot_point_estimate_fit(self, estimate='mean', time=None, data_err=None, fit_result=None,
                                 data_label=r'$D_t$', fit_label=r'$\hat{y}_t$',
                                 ylabel='Value', xlabel='Time', figsize=None,
                                 fit_color='red', legend_loc='upper center',
                                 legend_bbox_to_anchor=(0.5, 0.55), save_path=None, **kwargs):
        """Plot a point-estimate fit and its residuals in two panels.

        The top panel shows the training data (error bars if ``data_err`` is given,
        otherwise ``'+'`` markers) with the point-estimate fit overlaid; the bottom
        panel shows the residuals and shares the x-axis.

        Parameters
        ----------
        estimate : {'mean', 'median', 'mode'}, default 'mean'
            Type of point estimate. Ignored if ``fit_result`` is given.
        time : array-like, optional
            x-values of the training data. Defaults to ``np.arange(len(train_data))``.
        data_err : array-like or float, optional
            Uncertainties on the data, drawn as error bars. If None, the data are
            drawn as markers without error bars.
        fit_result : dict, optional
            Result of a previous ``point_estimate_forecast`` call to reuse instead of
            recomputing it. If None, ``point_estimate_forecast(estimate=estimate)`` is
            called.
        data_label : str, default LaTeX label for D_t
            Legend label of the data.
        fit_label : str, default LaTeX label for the fitted values
            Legend label of the fit.
        ylabel : str, default "Value"
            y-axis label of the top panel.
        xlabel : str, default "Time"
            x-axis label.
        figsize : tuple of float, optional
            Figure size in inches. Defaults to ``(fig_width, 2 * fig_height)``.
        fit_color : str, default "red"
            Colour of the fit line.
        legend_loc : str, default "upper center"
            Location of the figure legend.
        legend_bbox_to_anchor : tuple of float, default (0.5, 0.55)
            Anchor of the figure legend in figure coordinates.
        save_path : str, optional
            If given, also save the figure there as a PDF (300 dpi, tight bounding
            box, transparent background).
        **kwargs
            ``fig_width`` (default 6) and ``fig_height`` (default 3), used to build
            ``figsize`` when it is not given.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : numpy.ndarray of matplotlib.axes.Axes
            The two panels (data and fit, residuals).
        fit_result : dict
            The ``point_estimate_forecast`` result that was plotted.
        """
        fit_result = fit_result if fit_result is not None else self.point_estimate_forecast(estimate=estimate)
        y_fit, residuals = fit_result['y_fit'], fit_result['residuals']
        time = time if time is not None else np.arange(len(self.train_data))
        

        fig_width, fig_height = kwargs.get("fig_width", 6), kwargs.get("fig_height", 3)
        figsize = figsize if figsize is not None else (fig_width, 2 * fig_height)
        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if data_err is not None:
            ax[0].errorbar(time, self.train_data, yerr=data_err, fmt='o', c='black', ms=1, capsize=2, label=data_label)
        else:
            ax[0].plot(time, self.train_data, marker='+', ms=4, lw=0, c='black', label=data_label)
        ax[0].plot(time, y_fit, c=fit_color, lw=1, label=fit_label)
        ax[1].plot(time, residuals, c='black')

        ax[1].set_xlabel(xlabel, fontsize=10)
        ax[0].set_ylabel(ylabel)
        ax[1].set_ylabel(r'Residuals ($D_t - y_t$)')

        leg = fig.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, frameon=True, fontsize=7)
        frame = leg.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_alpha(1.0)

        ax[0].grid(alpha=0.5)
        ax[1].grid(alpha=0.5)
        fig.tight_layout(pad=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True, format="pdf")

        return fig, ax, fit_result

    def corner_plot(self, params=None, include_prior=False, prior_kwargs=None,
                     posterior_color='tomato', posterior_kinds='kde',
                     true_values=None, xlim=None, figsize=None, facecolor='w',
                     upper=False, tick_labelsize=7, legend=True,
                     legend_loc='lower center', save_path=None, **kwargs):
        """Draw a 2D posterior corner plot with ``anesthetic``.

        Parameters
        ----------
        params : list of str, optional
            Parameters to plot. Defaults to the AR/MA coefficients (or PACF alphas)
            plus ``sigma`` and ``mu``. ``init_y`` is left out by default because it
            is a nuisance parameter rather than a model parameter of interest; pass
            ``params`` explicitly to include it or to plot a different subset.
        include_prior : bool, default False
            If True, overlay the prior from ``self.chain.prior()``. Only available
            on a live ``NestedSamples`` chain, not on one reloaded from certain older
            CSVs.
        prior_kwargs : dict, optional
            Extra options for plotting the prior, forwarded to ``plot_2d``. Defaults
            to ``alpha=0.9``, ``color='grey'``, ``kinds=posterior_kinds`` and
            ``label='prior'``.
        posterior_color : str, default "tomato"
            Colour of the posterior.
        posterior_kinds : str or dict, default "kde"
            Plot kinds passed to ``plot_2d`` for the posterior (and, by default, the
            prior).
        true_values : dict, optional
            Mapping ``{param: value}`` (a subset of ``params`` is fine) drawn as
            black dashed reference lines. A "true values" legend entry is added via a
            proxy line on the bottom-left panel if the first parameter is included.
        xlim : dict, optional
            Mapping ``{param: (low, high)}`` of x-limits applied to that parameter's
            column of panels.
        figsize : tuple of float, optional
            Figure size in inches. Defaults to ``(fig_width, 1.5 * fig_height)``.
        facecolor : str, default "w"
            Figure face colour.
        upper : bool, default False
            Whether to create the 2D panels above the diagonal (passed to
            ``make_2d_axes``).
        tick_labelsize : int, default 7
            Font size of the tick labels.
        legend : bool, default True
            Whether to draw the "true values" legend.
        legend_loc : str, default "lower center"
            Location of that legend.
        save_path : str, optional
            If given, also save the figure there as a PDF (300 dpi, tight bounding
            box, transparent background).
        **kwargs
            ``fig_width`` (default 6) and ``fig_height`` (default 4), used to build
            ``figsize`` when it is not given.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        axes : anesthetic.plot.AxesDataFrame
            The grid of panel axes.

        Notes
        -----
        The panel indexing used for ``true_values`` and ``xlim`` mirrors
        ``anesthetic``'s lower-triangle (``upper=False``) grid layout. If you pass
        ``upper=True``, or a different ``anesthetic`` version changes that indexing,
        you may need to adjust which panels these are applied to.
        """
        if params is None:
            ar_keys, ma_keys, _ = self._arma_keys()
            params = ar_keys + ma_keys + ['sigma', 'mu']

        fig_width, fig_height = kwargs.get("fig_width", 6), kwargs.get("fig_height", 4)
        figsize = figsize if figsize is not None else (fig_width, 1.5 * fig_height)

        fig, axes = make_2d_axes(params, figsize=figsize, facecolor=facecolor, upper=upper)

        if include_prior:
            prior = self.chain.prior()
            prior_kwargs = dict(prior_kwargs) if prior_kwargs else {}
            prior_kwargs.setdefault('alpha', 0.9)
            prior_kwargs.setdefault('color', 'grey')
            prior_kwargs.setdefault('kinds', posterior_kinds)
            prior_kwargs.setdefault('label', 'prior')
            prior.plot_2d(axes, **prior_kwargs)

        self.chain.plot_2d(axes, alpha=0.9, color=posterior_color, kinds=posterior_kinds, label='posterior')

        if true_values:
            axes.axlines(true_values, c='black', linestyle='--', lw=0.7)
            first_param = params[0]
            if first_param in true_values:
                axes.iloc[-1, 0].axhline(y=true_values[first_param], c='black', linestyle='--',
                                          lw=0.7, label='true values')
                if legend:
                    axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)),
                                             loc=legend_loc, ncols=2)

        if xlim:
            params_list = list(params)
            for param, lims in xlim.items():
                axes.iloc[-1, params_list.index(param)].set_xlim(*lims)

        axes.tick_params(labelsize=tick_labelsize)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True, format="pdf")

        return fig, axes
