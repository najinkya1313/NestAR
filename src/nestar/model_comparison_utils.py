"""ARIMA model comparison by nested sampling.

Provides ``ARIMA_model_comparison``, which runs a grid of ARIMA(p, d, q)
nested-sampling fits and exposes the results (evidences and their errors, log
posteriors, rejection-sampling acceptance rate ``V``, maximum log-likelihood,
BIC, D0 Occam factor and net prior-volume effect) as plain attributes on the
instance. Also provides ``stationarity_test`` (ADF and KPSS tests) and
``compute_d0_occam_factor``.

This module owns the *sampling* logic and depends on ``jax``, ``ARIMA_ns``
and ``ARIMA_fast``. To only load and plot a results file produced by a run
elsewhere, use the sampler-free ``arima_results`` module instead.
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
            """Run ADF and KPSS stationarity tests on a time series and print the results.

            Runs the augmented Dickey-Fuller (ADF) test, with the lag length chosen by
            AIC, and the KPSS test with a constant regression and automatic lag
            selection. For each test the statistic, p-value, lags used and critical
            values are printed, followed by a one-line verdict at the 5% level.

            Parameters
            ----------
            timeseries : array-like
                One-dimensional time series to test.

            Returns
            -------
            tuple of None
                ``(None, None)``: both tests print their output rather than returning it.

            Notes
            -----
            The two tests have opposite null hypotheses. For ADF the null is a unit root
            (non-stationarity), so ``p < 0.05`` indicates a stationary series. For KPSS
            the null is stationarity, so ``p < 0.05`` indicates a non-stationary series.
            """

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
    """Approximate the Occam-factor contribution of the initial-value (D0) block.

    For each initial-value parameter ``init_y_i`` (``i = 1, ..., p``) this
    computes ``log(sigma_post / tau)``, the log of the ratio of its posterior
    standard deviation to its prior scale. A posterior much narrower than the
    prior gives a strongly negative value (a genuine Occam penalty); a posterior
    about as wide as the prior gives a value near zero (the parameter is
    essentially unconstrained by the data).

    Parameters
    ----------
    posterior_samples : pandas.DataFrame-like
        Posterior samples with columns ``init_y_1`` ... ``init_y_p``.
    order : tuple of int
        ARIMA order ``(p, d, q)``.
    tau : float
        Prior scale of the ``init_y`` parameters. It must equal the scale
        actually used for ``init_y`` in ``prior_parameters()``, which is currently
        ``mu_scale`` for both the ``'normal'`` and ``'pacf'`` priors, so callers
        should pass ``mu_scale``.

    Returns
    -------
    total : float
        Sum of the per-parameter contributions; 0.0 if ``p == 0``.
    per_param : dict of {str: float}
        Contribution of each parameter, keyed ``'init_y_1'``, ``'init_y_2'``,
        ...; empty if ``p == 0``, since there are then no D0 parameters to
        penalise.

    Notes
    -----
    This is an approximate, per-parameter diagnostic. It assumes near-Gaussian
    marginals and treats the ``init_y`` parameters as independent of each other
    and of ``phi``, ``theta``, ``sigma`` and ``mu``. It is not an exact
    decomposition of the evidence.
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

    Fits every model ``(p, d, q)`` with ``0 <= p <= max_p`` and ``0 <= q <= max_q``
    at a fixed differencing order ``d`` (excluding the trivial ``(0, d, 0)``
    model) using ``ARIMA_Nested_Sampler``, and stores the per-model results as
    plain attributes. Call ``run`` once to perform the grid search, or
    ``load_evidence_file`` to load the results of an earlier run without
    resampling. The results can then be plotted with ``plot_evidence_heatmap`` and
    ``compare`` and summarised with ``prior_volume_report``.

    Parameters
    ----------
    data : array-like
        One-dimensional time series to analyse.
    max_p : int
        Maximum AR order of the grid.
    max_q : int
        Maximum MA order of the grid.
    d : int
        Differencing order shared by every model in the grid.
    num_live : int
        Number of live points used by the nested sampler.
    num_delete : int
        Number of points deleted at each nested-sampling iteration.
    seed : int
        Base random seed. The model at position ``k`` in the grid is sampled with
        seed ``seed + k``.
    prior_type : {'normal', 'pacf', 'uniform'}, default 'normal'
        Prior on the ARMA coefficients, passed to the sampler. Any value other
        than ``'pacf'`` or ``'uniform'`` is treated as ``'normal'``.
    mu_mean : float, default 0
        Prior mean of the long-term mean of the data.
    mu_scale : float, default 1
        Prior scale of the long-term mean of the data. It doubles as ``tau``, the
        prior scale of the D0 (``init_y``) parameters; see
        ``compute_d0_occam_factor``.
    prior_scale : float, default 1
        Prior scale of the AR and MA coefficients (``phi``, ``theta``).
    inner_steps_factor : int, default 6
        Factor setting the number of inner steps taken by the sampler at each
        nested-sampling iteration; passed to ``ARIMA_Nested_Sampler``.
    file_name : str, optional
        Path of an evidence file. If given, the file is started afresh (with a
        warning first if it already exists) and, during ``run``, is written to in
        real time as each model finishes, so that an interrupted run does not lose
        its progress. Once the whole grid is done, a final set of columns
        (normalised posterior, BIC, D0 Occam factor and net prior-volume effect)
        is written; see ``run``.
    prior_bounds : dict, optional
        Prior bounds passed through to ``ARIMA_Nested_Sampler``. An empty
        dictionary is used if None.
    include_mean_param, include_scale_param : bool, default True
        Whether the parameter count used in the BIC,
        ``k = p + q + (1 if mean) + (1 if scale)``, includes a mean parameter
        and/or a noise-scale parameter. Check that this matches what
        ``ARIMA_Nested_Sampler`` actually fits; see ``num_arima_params`` in
        ``arima_results``.
    meas_sigma : array-like or float, optional
        Measurement uncertainties of the data, passed through to
        ``ARIMA_Nested_Sampler``.

    Attributes
    ----------
    orders : list of tuple of int
        ARIMA orders ``(p, d, q)`` of the models in the grid, ordered by ``p``
        and then by ``q``. Each of the following per-model attributes is ordered
        like ``orders``. All of them are None until populated by ``run`` or
        ``load_evidence_file``.
    evidences : numpy.ndarray
        Log evidence of each model.
    evidence_err : numpy.ndarray
        Uncertainty on each log evidence.
    log_posteriors : numpy.ndarray
        Log posterior probability of each model, normalised over the grid.
    V : numpy.ndarray
        Acceptance rate of the rejection sampling that restricts the prior for
        each model. It is 1.0 for ``prior_type='pacf'`` (no rejection is needed)
        and NaN for ``prior_type='uniform'`` (no restriction is applied).
    max_loglikelihood : numpy.ndarray
        Maximum log-likelihood found for each model.
    BIC : numpy.ndarray
        Bayesian information criterion of each model.
    d0_occam : numpy.ndarray
        Total D0 Occam factor of each model; see ``compute_d0_occam_factor``.
    d0_per_param : list of dict or None
        Per-parameter D0 Occam contributions of each model. Only available on a
        live object straight after ``run``; None after ``load_evidence_file``.
    log_V_boost : numpy.ndarray
        Log prior-volume boost of each model, derived from the acceptance rate
        ``V``.
    net_prior_volume_effect : numpy.ndarray
        Net prior-volume effect of each model: ``log_V_boost + d0_occam``.

    Notes
    -----
    If ``file_name`` is given, the file is emptied when the instance is
    constructed, and again at the start of ``run``, with a warning if it already
    exists. To read an existing evidence file without wiping it, construct the
    instance without ``file_name`` and pass the path to ``load_evidence_file``
    instead.

    Examples
    --------
    >>> comp = ARIMA_model_comparison(data, max_p=3, max_q=3, d=0,
    ...                               num_live=500, num_delete=100, seed=0,
    ...                               file_name="evidence.txt")
    >>> comp.run()
    >>> fig = comp.plot_evidence_heatmap("log_posteriors")
    """

    def __init__(self, data, max_p, max_q, d, num_live, num_delete, seed,
                 prior_type="normal", mu_mean=0, mu_scale=1, prior_scale=1,
                 inner_steps_factor=6, file_name=None, prior_bounds=None,
                 include_mean_param=True, include_scale_param=True,meas_sigma=None):
        """Store the configuration and, if ``file_name`` is given, start a fresh file.

        See the class docstring for a description of the parameters.
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
        """Run the ARIMA(p, d, q) grid search and populate the result attributes.

        Each model in the grid is fitted by nested sampling, and its log evidence,
        evidence error, acceptance rate ``V``, maximum log-likelihood, BIC and D0
        Occam factor are recorded. Progress is printed after each model. When all
        models are done, the log posteriors are normalised over the grid (and checked
        to sum to one) and the prior-volume quantities are derived.

        If ``self.file_name`` is set, the file is written in two stages:

        1. In real time, one line per model as soon as it finishes, with the fields
           ``Order``, ``Seed``, ``Evidence``, ``Error``, ``V``, ``MaxLogL``, ``BIC``,
           ``D0Occam`` and ``NetPriorVolume``. This protects against losing progress
           if the run is interrupted. ``NetPriorVolume`` can be written per line
           (unlike ``Posterior``) because it depends only on that model's own ``V``
           and D0 Occam factor, not on the evidences of the whole grid.
        2. Once the entire grid is done, the file is rewritten with a ``Posterior``
           field (the normalised log posterior) added to every line. This can only
           happen at the end because the posterior of any one model depends on the
           evidences of every other model in the grid.

        Returns
        -------
        ARIMA_model_comparison
            This instance, to allow chaining.

        Notes
        -----
        The file, if set, is emptied at the start of the run, so any earlier contents
        are lost.

        The ``Seed`` field written to the file is always the base seed ``self.seed``,
        not the seed actually used for that model (``self.seed`` plus the model's
        position in the grid).

        For ``prior_type='uniform'`` no acceptance rate ``V`` is available, so the
        prior-volume quantities of each model are not defined.
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
        """Load a saved evidence file into this instance without re-running the sampler.

        Equivalent to calling ``arima_results.load_evidence_file`` directly and
        unpacking the returned dictionary into this instance's attributes.

        Parameters
        ----------
        file_name : str, optional
            Path of the evidence file. If given, it replaces ``self.file_name``; if
            None, ``self.file_name`` is used.
        check_normalization : bool, default True
            Passed to ``arima_results.load_evidence_file``; if True, the loaded model
            posterior probabilities are checked for normalisation.

        Returns
        -------
        ARIMA_model_comparison
            This instance, to allow chaining.

        Raises
        ------
        ValueError
            If no ``file_name`` is given and none was set on the instance.

        Notes
        -----
        ``d0_per_param`` is set to None: the per-parameter D0 detail is not stored in
        the text file, only the per-model total (``d0_occam``). It is only available
        on a live object straight after ``run``.
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
        """Look up a stored quantity by name for plotting.

        Parameters
        ----------
        name : str
            One of ``'log_posteriors'``, ``'evidences'``, ``'BIC'``, ``'V'``,
            ``'max_loglikelihood'``, ``'d0_occam'``, ``'log_V_boost'`` or
            ``'net_prior_volume_effect'``.

        Returns
        -------
        values : numpy.ndarray
            The per-model values.
        errors : numpy.ndarray or None
            ``evidence_err`` for ``'log_posteriors'`` and ``'evidences'``; None for
            all other quantities.

        Raises
        ------
        ValueError
            If ``name`` is unknown, or if the requested quantity has not been
            populated (call ``run`` or ``load_evidence_file`` first).
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
            raise ValueError(f"self.{name} is not populated -- run() or load_evidence_file() first.")
        return values, errors

    def plot_evidence_heatmap(self, quantity="log_posteriors", **kwargs):
        """Plot a single heatmap of one quantity over the ``(p, q)`` grid.

        Parameters
        ----------
        quantity : str, default "log_posteriors"
            Name of the quantity to plot: ``'log_posteriors'``, ``'BIC'``, ``'V'``,
            ``'evidences'``, ``'max_loglikelihood'``, ``'d0_occam'``,
            ``'log_V_boost'`` or ``'net_prior_volume_effect'``.
        **kwargs
            Forwarded to ``arima_results.plot_evidence_heatmap``. ``invert`` defaults
            to True only for ``'BIC'`` (lower is better), and ``title`` and
            ``cbar_label`` default to the quantity name.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the heatmap.

        Raises
        ------
        ValueError
            If ``quantity`` is unknown or has not been populated.
        """
        data = self._quantity(quantity)
        kwargs.setdefault("invert", quantity == "BIC")  # lower BIC is better
        kwargs.setdefault("title", quantity)
        kwargs.setdefault("cbar_label", quantity)
        return ar.plot_evidence_heatmap(data, self.max_p, max_q=self.max_q,
                                         orders=self.orders, **kwargs)

    def compare(self, quantity1="log_posteriors", quantity2=None, labels=None, **kwargs):
        """Compare two quantities side by side on the ``(p, q)`` grid.

        If ``quantity2`` is None this falls back to the ordinary single heatmap of
        ``quantity1`` (same as ``plot_evidence_heatmap``).

        Parameters
        ----------
        quantity1 : str, default "log_posteriors"
            Name of the quantity for the left panel; see ``plot_evidence_heatmap``
            for the available names.
        quantity2 : str, optional
            Name of the quantity for the right panel. If None, only ``quantity1`` is
            plotted.
        labels : tuple of str, optional
            Panel titles and colourbar labels. Defaults to the two quantity names.
        **kwargs
            Forwarded to ``arima_results.plot_comparison_heatmap`` (or to
            ``plot_evidence_heatmap`` when ``quantity2`` is None). ``invert`` may be
            given as a per-panel tuple and defaults to True for any panel showing
            ``'BIC'``.

        Returns
        -------
        matplotlib.figure.Figure
            The comparison figure.

        Raises
        ------
        ValueError
            If either quantity is unknown or has not been populated.

        Examples
        --------
        >>> comp.compare("log_posteriors", "BIC")
        >>> comp.compare("log_posteriors", "V")
        >>> comp.compare("log_posteriors", "net_prior_volume_effect")
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
        """Split a log-posterior difference into prior-volume and likelihood parts.

        Quantifies how much of the raw log-posterior difference between two models is
        prior-volume bookkeeping (``net_prior_volume_effect``: the rejection-sampling
        renormalisation, ``log_V_boost``, plus the D0 Occam term, ``d0_occam``, which
        is a penalty when negative) and how much is likelihood-driven signal.

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
              delta, i.e. what is left once the prior-volume bookkeeping is backed
              out.
            - ``'fraction_of_raw_delta_from_prior_volume'`` : net prior-volume delta
              divided by the raw delta (NaN if the raw delta is zero).

        Raises
        ------
        ValueError
            If either order is not present in ``self.orders``.

        Examples
        --------
        To compare ARIMA(9, 0, 1) against ARIMA(0, 0, 1):

        >>> comp.prior_volume_report(order_of_interest=(9, 0, 1),
        ...                          baseline_order=(0, 0, 1))
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
