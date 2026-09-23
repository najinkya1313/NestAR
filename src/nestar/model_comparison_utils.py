"""
arima_model_comparison.py

ARIMAModelComparison: runs a grid of ARIMA(p, d, q) nested-sampling fits and
exposes the results (evidences, errors, log posteriors, acceptance rate V,
max log-likelihood, BIC, D0 Occam factor, net prior-volume effect) as plain
attributes on the instance.

This module owns the *sampling* logic and depends on jax / ARIMA_ns /
ARIMA_fast. See arima_results.py if you just want to load
and plot a results file that was produced by a run elsewhere.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.tsa.stattools import adfuller, kpss

from . import arima_results as ar
from .ARIMA import ARIMA_fast
from .ARIMA_ns import (
    ARIMA_Nested_Sampler,
    loglikelihood,
    prior_parameters,
)

##Stationarity test
def stationarity_test(timeseries):

            def adf_test(timeseries):
                print("Results of Dickey-Fuller Test:")
                dftest = adfuller(timeseries, autolag="AIC")
                dfoutput = pd.Series(
                    dftest[0:4],
                    index=[
                        "Test Statistic",
                        "p-value",
                        "#Lags Used",
                        "Number of Observations Used",
                    ],
                )
                for key, value in dftest[4].items():
                    dfoutput["Critical Value (%s)" % key] = value
                print(dfoutput)

                if dftest[1] < 0.05:
                    print("Series stationary acc to ADF Test")
                else:
                    print("Non stationarity acc to ADF test")


            def kpss_test(timeseries):
                print("Results of KPSS Test:")
                kpsstest = kpss(timeseries, regression="c", nlags="auto")
                kpss_output = pd.Series(
                    kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
                )
                for key, value in kpsstest[3].items():
                    kpss_output["Critical Value (%s)" % key] = value
                print(kpss_output)

                if kpss_output[1] < 0.05:
                    print("Series non-stationary acc to KPSS Test")
                else:
                    print("Series stationarity acc to KPSS test")

            return adf_test(timeseries), kpss_test(timeseries)



def compute_d0_occam_factor(posterior_samples, order, tau):
    """
    Approximate per-parameter Occam-factor contribution of the D0 (init_y)
    block: log(sigma_post / tau) for each init_y_i, using its posterior std
    against its prior scale tau. tau must equal the actual prior scale used
    for init_y in prior_parameters() -- currently that's mu_scale for both
    prior_type='normal' and 'pacf' (init_y_scale = mu_scale there), so
    callers should pass self.mu_scale.

    Same family of approximation as the D_KL already reported in Fig. 8:
    posterior much narrower than prior => strongly negative (real penalty);
    posterior ~= prior width => near zero (parameter along for the ride).
    This is an approximate, per-parameter diagnostic (assumes near-Gaussian
    marginals, treated independently of phi/theta/sigma/mu) -- not an exact
    evidence decomposition. Present it as such.

    Returns (total, per_param) where per_param is a dict {init_y_i: contrib}.
    Returns (0.0, {}) for p=0, since there are no D0 parameters to penalize.
    """
    p, d, q = order
    if p == 0:
        return 0.0, {}
    per_param = {}
    total = 0.0
    for i in range(p):
        key = f'init_y_{i+1}'
        sigma_post = float(posterior_samples[key].std())
        contrib = float(np.log(sigma_post / tau))
        per_param[key] = contrib
        total += contrib
    return total, per_param



class ARIMA_model_comparison:
    """Grid search over ARIMA(p, d, q) models via nested sampling.

    Call `.run()` once to perform the grid search;
    """

    def __init__(self, data, max_p, max_q, d, num_live, num_delete, seed,
                 prior_type="normal", mu_mean=0, mu_scale=1, prior_scale=1,
                 inner_steps_factor=6, file_name=None, prior_bounds=None,
                 include_mean_param=True, include_scale_param=True,meas_sigma=None):
        """
        data : time series data to be analyzed
        max_p : max AR order of the grid
        max_q : max MA order of the grid
        d : differencing order d of the ARIMA model
        num_live : number of live points to be used
        num_delete : number of points to delete at each iteration
        seed : random seed
        mu_mean, mu_scale : prior mean/scale of the long term mean of data.
            mu_scale doubles as tau, the D0 (init_y) prior scale -- see
            compute_d0_occam_factor.
        prior_scale : prior scale for the AR and MA coefficients (phi, theta)
        file_name : if given, each call to `.run()` starts this file fresh
            (warns first if it already exists, so you don't silently
            overwrite a previous run) and then writes to it in real time as
            each model finishes, so a crash mid-run doesn't lose progress.
            A final normalized-posterior/BIC/D0/net-prior-volume column set
            is added once the whole grid is done -- see `.run()`.
        include_mean_param, include_scale_param : whether the BIC parameter
            count k = p + q + (1 if mean) + (1 if scale) should include a
            mean and/or noise-scale parameter. Check this matches what
            ARIMA_Nested_Sampler actually fits -- see num_arima_params().
        """
        self.data = data
        self.max_p = max_p
        self.max_q = max_q
        self.d = d
        self.num_live = num_live
        self.num_delete = num_delete
        self.seed = seed
        self.prior_type = prior_type
        self.mu_mean = mu_mean
        self.mu_scale = mu_scale
        self.prior_scale = prior_scale
        self.inner_steps_factor = inner_steps_factor
        self.file_name = file_name
        self.prior_bounds = prior_bounds if prior_bounds is not None else {}
        self.include_mean_param = include_mean_param
        self.include_scale_param = include_scale_param
        self.meas_sigma = meas_sigma

        # populated by .run() (or by .load_evidence_file())
        self.orders = None
        self.evidences = None
        self.evidence_err = None
        self.log_posteriors = None
        self.V = None
        self.max_loglikelihood = None
        self.BIC = None
        self.d0_occam = None
        self.d0_per_param = None
        self.log_V_boost = None
        self.net_prior_volume_effect = None

        if self.file_name:
                    if os.path.exists(self.file_name):
                        warnings.warn(
                            f"'{self.file_name}' already exists and will be overwritten by this run. "
                            "If you meant to keep it, rename/move it (or change file_name) before calling .run().",
                            stacklevel=2,
                        )
                    open(self.file_name, "w").close()  # start from a clean file -- never append to stale/old data

        
        

    # ------------------------------------------------------------------ #
    # running the grid search
    # ------------------------------------------------------------------ #

    def run(self):
        """Run the ARIMA(p, d, q) grid search and populate self.* attributes.

        Writes to self.file_name (if set) in two stages:
          1. In real time, one line per model as soon as it finishes
             (Order, Seed, Evidence, Error, V, MaxLogL, BIC, D0Occam,
             NetPriorVolume) -- protects against losing progress if the run
             is interrupted. NetPriorVolume can be written per-line (unlike
             Posterior) because it only depends on that single model's V and
             D0Occam, not on the whole grid's evidences.
          2. Once the *entire* grid is done, the file is rewritten with a
             Posterior= column added to every line. This can only happen at
             the end because the normalized posterior of any one model
             depends on the evidences of every other model in the grid.
        """
        evidences, evidence_err, V, max_loglikelihood, BIC = [], [], [], [], []
        d0_occam, d0_per_param = [], []
        orders = [(p, self.d, q) for p in range(self.max_p + 1)
                  for q in range(self.max_q + 1)]
        orders.remove((0, self.d, 0))  # removing the trivial case
        seeds = self.seed + np.arange(len(orders))
        n_data = len(self.data)

        if self.file_name:
            if os.path.exists(self.file_name):
                warnings.warn(
                    f"'{self.file_name}' already exists and will be overwritten by this run. ",
                    stacklevel=2,
                )
            open(self.file_name, "w").close()  # start from a clean file -- never append to stale/old data

        for order, seed in zip(orders, seeds):
            if self.prior_type == "pacf":
                model = ARIMA_Nested_Sampler(
                    data=self.data, order=order, mu_mean=self.mu_mean, mu_scale=self.mu_scale,
                    num_live=self.num_live, num_delete=self.num_delete, seed=seed,
                    prior_type="pacf", prior_bounds=self.prior_bounds,
                    inner_steps_factor=self.inner_steps_factor, prior_scale=self.prior_scale,meas_sigma=self.meas_sigma)
            elif self.prior_type == "uniform":
                model = ARIMA_Nested_Sampler(
                    data=self.data, order=order, mu_mean=self.mu_mean, mu_scale=self.mu_scale,
                    num_live=self.num_live, num_delete=self.num_delete, seed=seed,
                    prior_type="uniform", prior_bounds=self.prior_bounds,
                    inner_steps_factor=self.inner_steps_factor, prior_scale=self.prior_scale,meas_sigma=self.meas_sigma)
            else:
                model = ARIMA_Nested_Sampler(
                    data=self.data, order=order, mu_mean=self.mu_mean, mu_scale=self.mu_scale,
                    num_live=self.num_live, num_delete=self.num_delete, seed=seed,
                    prior_type="normal", prior_bounds=self.prior_bounds,
                    inner_steps_factor=self.inner_steps_factor, prior_scale=self.prior_scale,meas_sigma=self.meas_sigma)

            max_logL = float(np.max(model.posterior_samples.logL))
            num_params = ar.num_arima_params(order, include_mean=self.include_mean_param,
                                              include_scale=self.include_scale_param)
            bic = ar.compute_bic(max_logL, num_params, n_data)

            d0_total, d0_detail = compute_d0_occam_factor(
                model.posterior_samples, order, self.mu_scale)

            # model.V is 1.0 for prior_type='pacf' (no rejection loop -- every
            # draw in the (-1,1) box is valid by construction) and None for
            # prior_type='uniform' (unconstrained box prior, no S-restriction
            # at all -- see the flag on that branch in ARIMA_Nested_Sampler).
            v_this = model.V
            log_v_boost_this = ar.compute_log_V_boost(np.array([v_this]))[0] if v_this is not None else np.nan
            net_this = log_v_boost_this + d0_total if v_this is not None else np.nan

            evidences.append(model.log_evidence)
            evidence_err.append(model.log_evidence_err)
            V.append(v_this)
            max_loglikelihood.append(max_logL)
            BIC.append(bic)
            d0_occam.append(d0_total)
            d0_per_param.append(d0_detail)

            evidence_arr = np.array(evidences)
            best_idx = int(np.argmax(evidence_arr))
            print("----------------------x-------------------x---------------------x------")
            print(f"Evidence for {order} : {model.log_evidence} ; Error : {model.log_evidence_err}")
            print(f"V : {v_this} ; D0 Occam : {d0_total:.3f} ; Net prior-volume effect : {net_this:.3f}")
            print(f"Highest Evidence so far : {evidence_arr[best_idx]} for order : {orders[best_idx]}")
            print("----------------------x-------------------x----------------------x-----")

            if self.file_name:
                with open(self.file_name, "a") as f:
                    f.write(f"Order={order}, Seed={self.seed}, Evidence={model.log_evidence}, "
                            f"Error={model.log_evidence_err}, V={v_this}, "
                            f"MaxLogL={max_logL}, BIC={bic}, D0Occam={d0_total}, "
                            f"NetPriorVolume={net_this}\n")

        evidences = np.array(evidences)
        evidence_err = np.array(evidence_err)
        V = np.array(V, dtype=float)
        max_loglikelihood = np.array(max_loglikelihood)
        BIC = np.array(BIC)
        d0_occam = np.array(d0_occam)

        log_posteriors = evidences - logsumexp(evidences)
        ar.check_posteriors_sum_to_one(log_posteriors)

        log_V_boost = ar.compute_log_V_boost(V)
        net_prior_volume_effect = log_V_boost + d0_occam

        self.orders = orders
        self.evidences = evidences
        self.evidence_err = evidence_err
        self.log_posteriors = log_posteriors
        self.V = V
        self.max_loglikelihood = max_loglikelihood
        self.BIC = BIC
        self.d0_occam = d0_occam
        self.d0_per_param = d0_per_param
        self.log_V_boost = log_V_boost
        self.net_prior_volume_effect = net_prior_volume_effect

        if self.file_name:
            with open(self.file_name, "w") as f:
                for i, order in enumerate(orders):
                    f.write(f"Order={order}, Seed={self.seed}, Evidence={evidences[i]}, "
                            f"Error={evidence_err[i]}, V={V[i]}, MaxLogL={max_loglikelihood[i]}, "
                            f"BIC={BIC[i]}, D0Occam={d0_occam[i]}, "
                            f"NetPriorVolume={net_prior_volume_effect[i]}, "
                            f"Posterior={log_posteriors[i]}\n")

        return self

    # ------------------------------------------------------------------ #
    # thin wrappers around arima_results.py -- use those functions directly
    # if you don't need/want a class instance
    # ------------------------------------------------------------------ #

    def load_evidence_file(self, file_name=None, check_normalization=True):
        """Load a previously-saved evidence file into this instance's
        attributes (self.orders, self.evidences, ... self.net_prior_volume_effect),
        without re-running the sampler. Equivalent to calling
        arima_results.load_evidence_file() directly and unpacking the dict.

        Note: self.d0_per_param is set to None here -- per-parameter D0
        detail is not persisted to the text file, only the per-model total
        (self.d0_occam). It's only available on a live object right after
        .run().
        """
        file_name = file_name or self.file_name
        if file_name is None:
            raise ValueError("No file_name given and none was set on this instance")

        results = ar.load_evidence_file(file_name, check_normalization=check_normalization)
        self.orders = results["orders"]
        self.evidences = results["evidences"]
        self.evidence_err = results["evidence_err"]
        self.log_posteriors = results["log_posteriors"]
        self.V = results["V"]
        self.max_loglikelihood = results["max_loglikelihood"]
        self.BIC = results["BIC"]
        self.d0_occam = results["d0_occam"]
        self.d0_per_param = None
        self.log_V_boost = results["log_V_boost"]
        self.net_prior_volume_effect = results["net_prior_volume_effect"]
        self.file_name = file_name
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
            raise ValueError(f"self.{name} is not populated -- run() or load_evidence_file() first.")
        return values, errors

    def plot_evidence_heatmap(self, quantity="log_posteriors", **kwargs):
        """Ordinary single heatmap of one quantity ('log_posteriors', 'BIC',
        'V', 'evidences', 'max_loglikelihood', 'd0_occam', 'log_V_boost', or
        'net_prior_volume_effect') over the (p, q) grid.
        """
        data = self._quantity(quantity)
        kwargs.setdefault("invert", quantity == "BIC")  # lower BIC is better
        kwargs.setdefault("title", quantity)
        kwargs.setdefault("cbar_label", quantity)
        return ar.plot_evidence_heatmap(data, self.max_p, max_q=self.max_q,
                                         orders=self.orders, **kwargs)

    def compare(self, quantity1="log_posteriors", quantity2=None, labels=None, **kwargs):
        """Compare two quantities side by side, e.g.
            comp.compare("log_posteriors", "BIC")
            comp.compare("log_posteriors", "V")
            comp.compare("log_posteriors", "net_prior_volume_effect")
        If quantity2 is None, this just falls back to the ordinary single
        heatmap of quantity1 (same as plot_evidence_heatmap).
        """
        if quantity2 is None:
            return self.plot_evidence_heatmap(quantity1, **kwargs)

        data1 = self._quantity(quantity1)
        data2 = self._quantity(quantity2)
        labels = labels or (quantity1, quantity2)
        invert = kwargs.pop("invert", (quantity1 == "BIC", quantity2 == "BIC"))
        return ar.plot_comparison_heatmap(data1, data2, self.max_p, max_q=self.max_q,
                                           orders=self.orders, labels=labels,
                                           invert=invert, **kwargs)

    # ------------------------------------------------------------------ #
    # item 3c: quantifying V and D0 together, for a specific pair of orders
    # ------------------------------------------------------------------ #

    def prior_volume_report(self, order_of_interest, baseline_order):
        """Answers reviewer item 3c directly: how much of the raw log
        posterior difference between two orders is prior-volume bookkeeping
        (the rejection-sampling renormalisation minus the D0 Occam penalty),
        versus likelihood-driven signal.

        Example (sunspot grid, item 3c's exact question):
            comp.prior_volume_report(order_of_interest=(9, 0, 1),
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
