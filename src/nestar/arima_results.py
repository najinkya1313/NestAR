"""
arima_results.py

Everything to do with the *output* of an ARIMA nested-sampling run, whether
that's a model-comparison grid (an evidence text file) or a single model's
posterior chain (an anesthetic-format samples CSV). Two classes:

    EvidenceResults  -- self-consistent read + plot interface for a saved
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

Everything below the two classes' module-level helpers is plain
numpy/scipy/matplotlib and has no jax/anesthetic dependency; the classes
themselves do (EvidenceResults doesn't need jax/anesthetic at all, but
since this is now one file, importing it pulls those in regardless of
which class you actually use -- see the note in the module docstring
history below if you want to split them back apart).

This module is imported as `arima_results` (unchanged) by
arima_model_comparison.py (`import arima_results as ar`) and as
`from arima_results import PosteriorResults` by ARIMA_ns.py.
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
    """Number of free parameters k for an ARIMA(p, d, q) model, for use in BIC.

    By default this counts the p AR coefficients + q MA coefficients, plus one
    parameter for the long-term mean (mu) and one for the noise scale (sigma),
    since that's what ARIMA_Nested_Sampler appears to fit. Differencing order
    d does not add a free parameter.
    """
    p, d, q = order
    k = 2 * p + q
    if include_mean:
        k += 1
    if include_scale:
        k += 1
    return k


def compute_bic(max_loglikelihood, num_params, num_data):
    """BIC = k*ln(n) - 2*ln(L_max).

    Arguments:
    max_loglikelihood : the maximum log-likelihood attained by the model (a
        float, NOT an index -- see note in ARIMAModelComparison.run about a
        bug this fixes).
    num_params : number of free parameters k (see num_arima_params).
    num_data : number of data points n used to fit the model.
    """
    return num_params * np.log(num_data) - 2 * max_loglikelihood


# --------------------------------------------------------------------------- #
# Prior-volume bookkeeping (reviewer items 2, 3b, 3c)
# --------------------------------------------------------------------------- #

def compute_log_V_boost(V):
    """-log(V), the evidence boost from renormalising the rejection-sampled
    prior onto the stationary/invertible region S (see reviewer item 2).
    Grows with p+q as V (the rejection-sampling acceptance rate) shrinks.

    Derived purely from V, so it is NOT stored as its own column in the
    evidence file -- it's recomputed here on load, from whatever V column
    is present. Returns None if V is None. A V of exactly 0 (never accepted
    a single point -- shouldn't happen in practice for a converged run) maps
    to +inf rather than raising, so a degenerate cell doesn't crash a whole
    grid's plotting.
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
    """Split on `sep`, but ignore any `sep` found inside parentheses -- needed
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
    """Read back the results written by ARIMAModelComparison.run() (or by the
    old standalone ARIMA_model_comparison function).

    Understands any subset of the fields Order, Seed, Evidence, Error, V,
    MaxLogL, BIC, D0Occam, NetPriorVolume, Posterior that happen to be
    present on each line, so it is backward compatible with older evidence
    files that only have Order/Seed/Evidence/Error (any missing field comes
    back as None). If a file has no Posterior column, the log posteriors are
    (re)computed here from the Evidence column. log_V_boost is never read
    from the file directly -- it's always rederived from V via
    compute_log_V_boost, since it's a pure function of V and storing it
    separately would risk it going stale relative to V.

    Returns a dict with keys:
        orders                   : list of (p, d, q) tuples, or None if not present
        evidences                : np.ndarray of log evidences
        evidence_err              : np.ndarray of log evidence errors
        log_posteriors            : np.ndarray of normalized log posterior probabilities
        V                         : np.ndarray of rejection-sampling acceptance rates, or None
        max_loglikelihood         : np.ndarray of per-model max log-likelihoods, or None
        BIC                       : np.ndarray of per-model BIC values, or None
        d0_occam                  : np.ndarray of per-model D0 Occam-factor totals, or None
        log_V_boost                : np.ndarray of -log(V) per model, or None (derived from V)
        net_prior_volume_effect   : np.ndarray of log_V_boost + d0_occam, or None
            (read from a NetPriorVolume column if present; else recomputed
            from log_V_boost + d0_occam if both are available; else None)

    Note: per-parameter D0 detail (which init_y_i contributed how much) is
    NOT persisted to the text file -- only the per-model total. That detail
    only exists on a live ARIMA_model_comparison.d0_per_param right after
    .run(), not after a reload via .load_evidence_file().
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
    """Warn (never raise) if exp(log_posteriors) doesn't sum to ~1."""
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
    """Plot a heatmap of a per-model quantity (log posterior, BIC, V,
    d0_occam, net_prior_volume_effect, ...) on the (p, q) grid.

    Arguments:
    data : tuple (values, errors). `values` is either
        - a flat array ordered like the (p, q) grid excluding (0, d, 0)
          (the historical behaviour), used together with `orders`, or
        - already a 2D (max_q+1, max_p+1) array.
        `errors` may be None if the quantity has no associated uncertainty
        (e.g. BIC, V, d0_occam) -- annotations then omit the "+/-" line.
    max_p, max_q : max AR / MA order of the grid. max_q defaults to max_p
        (square grid), matching the old behaviour.
    orders : list of (p, d, q) tuples matching `values`, required when
        `values` is a flat array (i.e. whenever data didn't come pre-gridded).
    invert : invert the colormap -- use this for quantities where *lower* is
        better, e.g. BIC.
    title, cbar_label : optional axis title / colorbar label overrides.
    value_fmt : format string used for the annotated value (not the error).
    cbar_label_top : if True, draw cbar_label as a title above the colorbar
        (one size smaller than usual) instead of a side label -- avoids the
        side label crowding a neighbouring subplot's y-axis in a 1x2 layout.
        Used by plot_comparison_heatmap; plain single-panel calls default to
        the old side-label placement.
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
    """Side-by-side (1x2) comparison of two per-model quantities on the same
    (p, q) grid -- e.g. log posterior vs. BIC, log posterior vs. V, or
    log posterior vs. net_prior_volume_effect.

    `data1` / `data2` are each a (values, errors) tuple, exactly like the
    `data` argument to plot_evidence_heatmap. If one of them is None, this
    falls back to a single ordinary heatmap of the other.

    fontsize, contrast : per-panel tuples (panel0, panel1), forwarded to
        each plot_evidence_heatmap call as annotate_fontsize / contrast.
        contrast also accepts a bare scalar as shorthand for (x, x).
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
    """Flat forecast at the training-period mean."""
    return np.full(n_forecast, float(np.mean(np.asarray(train_data))))


def persistence_forecast(train_data, n_forecast):
    """Flat forecast at the last training-period value."""
    return np.full(n_forecast, float(np.asarray(train_data)[-1]))


def forecast_metrics(y_true, y_pred, sigma_pred=None):
    """RMSE, MAE always returned; LPD only if a per-step (or constant)
    sigma is given -- sigma_pred can be a scalar or an array matching y_true."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    metrics = {'RMSE': float(np.sqrt(np.mean(resid**2))), 'MAE': float(np.mean(np.abs(resid)))}
    if sigma_pred is not None:
        sigma_pred = np.broadcast_to(np.asarray(sigma_pred, dtype=float), y_true.shape)
        metrics['LPD'] = float(np.sum(-0.5*np.log(2*np.pi*sigma_pred**2) - 0.5*resid**2/sigma_pred**2))
    return metrics


def skill_score(metric_model, metric_baseline):
    """1 - model/baseline"""
    return 1.0 - metric_model/metric_baseline if metric_baseline != 0 else np.nan


# --------------------------------------------------------------------------- #
# EvidenceResults
# --------------------------------------------------------------------------- #

class EvidenceResults:
    """Self-consistent read + plot interface for a saved ARIMA
    model-comparison evidence file (the text file written by
    ARIMA_model_comparison.run()).

    This is the read-only, no-sampler counterpart to ARIMA_model_comparison:
    it exposes the same .plot_evidence_heatmap() / .compare() /
    .prior_volume_report() interface that class offers once you've called
    .load_evidence_file() on it, but as its own object built straight from
    the file, with no jax/ARIMA_ns dependency and no ability (or need) to
    run the grid itself.
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
        """(Re)read self.file_name (or a new file_name, if given) into this
        instance's attributes. Useful if the underlying file has since been
        updated -- e.g. a run() still in progress elsewhere, writing new
        lines as each model finishes."""
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
        """Map a quantity name to a (values, errors) tuple for plotting."""
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
        """Ordinary single heatmap of one quantity ('log_posteriors', 'BIC',
        'V', 'evidences', 'max_loglikelihood', 'd0_occam', 'log_V_boost', or
        'net_prior_volume_effect') over the (p, q) grid.
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
        """Compare two quantities side by side, e.g.
            results.compare("log_posteriors", "BIC")
            results.compare("log_posteriors", "V")
            results.compare("log_posteriors", "net_prior_volume_effect")
        If quantity2 is None, this just falls back to the ordinary single
        heatmap of quantity1 (same as plot_evidence_heatmap).

        quantity1 / quantity2 may each be a known quantity name (str,
        looked up via self._quantity()) or a raw array of custom values to
        plot directly (wrapped as (values, None), i.e. with no error bars).
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
        """Answers reviewer item 3c directly: how much of the raw log
        posterior difference between two orders is prior-volume bookkeeping
        (the rejection-sampling renormalisation minus the D0 Occam penalty),
        versus likelihood-driven signal.

        Example (sunspot grid, item 3c's exact question):
            results.prior_volume_report(order_of_interest=(9, 0, 1),
                                         baseline_order=(0, 0, 1))

        Returns a dict with the raw log-posterior delta, the net
        prior-volume-effect delta, and the "likelihood-only" delta obtained
        by subtracting the latter from the former -- i.e. what's left once
        prior-volume bookkeeping is backed out. Also reports what fraction
        of the raw delta the prior-volume term accounts for.
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
        """Answers reviewer item 7a: the cumulative posterior probability
        mass carried by the top-n models by log_posteriors, rather than
        reporting only the argmax.

        Example (sunspot grid, item 7a's suggested phrasing):
            results.cumulative_posterior_mass(1)
            # -> {'individual_mass': array([0.17...]), 'cumulative_mass': 0.17...}
            results.cumulative_posterior_mass(5)
            # -> cumulative_mass just over 0.5

        n : how many top-ranked models to include. Clipped to the number
            of models in the grid if larger.

        Returns a dict:
            n                : n, after clipping
            top_orders       : the top-n (p, d, q) orders, best first, or
                                None if orders weren't loaded from the file
            individual_mass  : posterior probability of each of those n
                                models individually (exp(log_posteriors)),
                                best first
            cumulative_mass  : sum of individual_mass -- the total
                                posterior probability carried by the top n
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
    """Counts phi_N/theta_N (normal prior) or alpha_ar_N/alpha_ma_N (pacf
    prior) columns on the chain to recover p, q, and prior_type. d is never
    inferred -- it's not a sampled parameter, so it leaves no trace in the
    chain's columns."""
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
    if prior_type == 'pacf':
        ar_block, ma_block = jnp.asarray(ar_block), jnp.asarray(ma_block)
        phi = pacf_to_arma(ar_block) if ar_block.shape[0] else jnp.array([])
        theta = -pacf_to_arma(ma_block) if ma_block.shape[0] else jnp.array([])
        return phi, theta
    return jnp.asarray(ar_block), jnp.asarray(ma_block)


def _future_time_index(overall_time, upper_index, num_forecast):
    """overall_time[upper_index:upper_index+num_forecast] when those future
    timestamps exist; otherwise extrapolates forward from the last timestamp
    at the (assumed constant) step of overall_time -- covers the no-test-data
    case where overall_time only spans the training period."""
    overall_time = np.asarray(overall_time)
    if upper_index + num_forecast <= len(overall_time):
        return overall_time[upper_index:upper_index + num_forecast]
    step = overall_time[1] - overall_time[0] if len(overall_time) > 1 else 1
    start = overall_time[-1] + step
    return start + step * np.arange(num_forecast)


def _weighted_mode(values, weights, grid_size=2000):
    """1D weighted-KDE mode estimate: fits a Gaussian KDE to `values` with
    per-sample `weights` and returns the grid location of its peak. This is
    a marginal (per-parameter) mode, not a joint/MAP estimate. Falls back to
    the weighted mean if a KDE can't be built (e.g. zero-variance samples).
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
    """Self-consistent read + analysis interface for a saved ARIMA posterior
    chain (an anesthetic-format CSV written by NestedSamples.to_csv(), as
    produced by ARIMA_Nested_Sampler / ARIMA_model_comparison). Initialized
    from either that CSV's path or an already-loaded chain object (e.g. a
    live NestedSamples instance).

    p, q, and prior_type are inferred from the chain's columns if not given;
    d is not inferable from the chain and defaults to 0.

    Two families of methods:
      - full-chain forecasting (unchanged from the former ARIMAForecaster):
        compute_forecast / outsample_forecast propagate the whole posterior
        through fgivenx for an out-of-sample forecast with proper posterior
        uncertainty bands; insample_forecast does the same for the in-sample
        fit + residuals.
      - point-estimate analysis (new): posterior_point_estimate collapses
        the chain to a single mean/median/mode value per parameter;
        point_estimate_forecast uses that single point to compute an
        in-sample fit (and, optionally, a deterministic forward forecast)
        directly via ARIMA_fast/ARIMA_forecast, with no posterior
        propagation; plot_point_estimate_fit plots that fit against the data
        with residuals; corner_plot draws the 2D posterior (optionally with
        the prior overlaid).
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
        """(ar_keys, ma_keys, init_y_keys) column names for this object's
        order/prior_type -- ar/ma keys are phi_i/theta_j for prior_type
        'normal' or alpha_ar_i/alpha_ma_j for 'pacf'."""
        p, d, q = self.order
        ar_keys = [f'alpha_ar_{i+1}' for i in range(p)] if self.prior_type == 'pacf' else [f'phi_{i+1}' for i in range(p)]
        ma_keys = [f'alpha_ma_{j+1}' for j in range(q)] if self.prior_type == 'pacf' else [f'theta_{j+1}' for j in range(q)]
        init_y_keys = [f'init_y_{i+1}' for i in range(p)]
        return ar_keys, ma_keys, init_y_keys

    def _param_keys(self):
        """Full flat list of sampled-parameter column names: ar + ma +
        sigma + mu + init_y."""
        ar_keys, ma_keys, init_y_keys = self._arma_keys()
        return ar_keys + ma_keys + ['sigma', 'mu'] + init_y_keys

    # ------------------------------------------------------------------ #
    # full-chain (posterior-propagated) forecasting
    # ------------------------------------------------------------------ #

    def _pack_samples(self, n_samples, seed=None):
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
        p, d, q = self.order
        params = jnp.asarray(params)
        ar_raw, ma_raw = params[0:p], params[p:p+q]
        sigma, mu = params[p+q], params[p+q+1]
        init_y = params[p+q+2:p+q+2+p]
        seed_i = int(params[-1])
        phi, theta = _get_arma_coeffs(ar_raw, ma_raw, self.prior_type)
        return phi, theta, sigma, mu, init_y, seed_i

    def _forecast_func(self, num_forecast):
        def f(x, params):
            phi, theta, sigma, mu, init_y, seed_i = self._unpack_params(params)
            return ARIMA_forecast(self.train_data, self.order, sigma, mu, phi, theta,
                                   num_forecast, init_y, seed_i)
        return f

    def _fit_func(self,deterministic=True):
        def f(x, params):
            phi, theta, sigma, mu, init_y, seed_i = self._unpack_params(params)
            if deterministic==True:
             return ARIMA_fast(self.train_data, self.order, 0.0, mu, phi, theta, init_y, seed_i)
            else:
             return ARIMA_fast(self.train_data, self.order,sigma, mu, phi, theta, init_y, seed_i)

        return f

    def _residual_func(self,deterministic=True):
        fit_func = self._fit_func(deterministic=deterministic)
        def f(x, params):
            return jnp.asarray(self.train_data) - fit_func(x, params)
        return f

    def compute_forecast(self, num_forecast, n_samples=1000, seed=0):
        """Draws n_samples from the chain, replays each through
        ARIMA_forecast, returns the forecast matrix and summary stats."""
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
     """
     

     label_fontsize, tick_labelsize, legend_fontsize, title_fontsize : font
        sizes for axis labels, tick labels, legend text, and title,
        matching the small-font style used elsewhere in the paper's figures
        (defaults: 9/7/7/9).
     show_legend, show_title : set False to omit either entirely -- e.g. the
        manual sunspot-forecast script omits both the legend() call and any
        title.
     cbar_labelsize, cbar_tick_position : forwarded to the fgivenx colorbar's
        tick_params(labelsize=...) and yaxis.set_ticks_position(...).
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
        """In-sample fit + residuals: plots posterior fit lines and residual
        lines against the training data, using fgivenx.plot_lines."""
        packed_samples = self._pack_samples(n_samples, seed=seed)
        fit_func, residual_func = self._fit_func(deterministic=deterministic), self._residual_func()

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

        estimate : 'mean', 'median', or 'mode'.
            'mean'/'median' use the chain's own weight-aware .mean()/.median().
            'mode' is a per-parameter (marginal) weighted-KDE mode -- not a
            joint/MAP estimate -- via _weighted_mode().
        param_keys : which chain columns to summarise; defaults to this
            object's full parameter set (ar/ma coeffs or pacf alphas, sigma,
            mu, init_y).

        Returns a dict {column_name: point_estimate_value}.
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
     """
     Computes D_t - y_hat_t across n_samples posterior draws, returned as an
     (n_samples, n_train) array -- the actual numbers behind the residual
     fan plotted by insample_forecast() (which only ever passes
     residual_func to fgivenx.plot_lines and never keeps the array).
     No time axis needed: residual_func's x argument only exists to satisfy
     fgivenx's calling convention and is never used in the computation itself.
     """
     packed_samples = packed_samples if packed_samples is not None else self._pack_samples(n_samples, seed=seed)
     residual_func = self._residual_func(deterministic=deterministic)
     residual_matrix = np.array([
        np.asarray(residual_func(None, p)) for p in tqdm.tqdm(packed_samples, desc="Pooled residuals")
    ])
     return residual_matrix

    def point_estimate_forecast(self, estimate='mean', num_forecast=0, deterministic=False,point_params=None, seed=None):
        """In-sample fit (and, optionally, a deterministic num_forecast-step
        forward extension) from a *single* point estimate of the posterior,
        rather than the full-chain propagation compute_forecast()/
        outsample_forecast() do. This is the point-estimate analogue of
        those two methods, generalizing the pattern of building a fit
        directly from posterior_means via ARIMA_fast to any of mean/median/
        mode, and adding an optional ARIMA_forecast extension.

        point_params : precomputed point estimate dict (e.g. from a previous
            call to posterior_point_estimate()) to reuse instead of
            recomputing; if None, calls posterior_point_estimate(estimate=estimate).

        Returns a dict:
            point_params : the point-estimate dict used, plus resolved 'phi'
                and 'theta' arrays (pacf alphas already transformed)
            y_fit        : in-sample fitted values, same length as train_data
            residuals    : train_data - y_fit
            y_forecast   : num_forecast-step-ahead point forecast, or None
                if num_forecast == 0
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
        """Two-panel (data+fit / residuals) plot from a point-estimate fit --
        top panel is the training data (errorbar if data_err is given, else
        a scatter of '+' markers) with the point-estimate fit overlaid;
        bottom panel is the residuals, sharing the x-axis.

        fit_result : reuse an already-computed point_estimate_forecast()
            result instead of recomputing it; if None, calls
            point_estimate_forecast(estimate=estimate).
        save_path : if given, saves the figure there (PDF, tight bbox,
            transparent background, 300 dpi) in addition to returning it.
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
        """2D posterior corner plot via anesthetic's make_2d_axes/plot_2d.

        params defaults to this object's AR/MA coefficients (or pacf alphas)
        plus sigma and mu -- NOT init_y, which is a nuisance parameter
        rather than a model parameter of interest by default; pass params=
        explicitly to include it, or to plot a different subset entirely.
        include_prior overlays self.chain.prior() (only available on a live
        NestedSamples chain, not one reloaded from certain older CSVs).
        true_values, if given, is a dict {param: value} (a subset of params
        is fine) drawn as black dashed reference lines, with a legend entry
        added via a proxy line on the bottom-left panel.
        xlim, if given, is a dict {param: (low, high)} applied to that
        param's column of panels.

        Note: the true_values/xlim panel-indexing here mirrors anesthetic's
        lower-triangle (upper=False) grid layout; if you pass upper=True or
        a different anesthetic version changes that indexing, you may need
        to adjust which panel these are applied to.
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
