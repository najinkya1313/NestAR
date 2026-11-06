"""
arima_model_comparison.py

Compare non-seasonal ARIMA(p, d, q) models using Nested Sampling.

The module runs a grid of ARIMA nested-sampling fits and exposes the
resulting evidences, evidence errors, log posterior probabilities,
acceptance rate V, maximum log-likelihoods, BIC values, D0 Occam
factors, and net prior-volume effects as attributes of an
ARIMA_model_comparison instance.

Optionally, when ``maximum_likelihood=True``, the maximum
log-likelihood of each model is refined using BFGS optimisation
(see ``arima_mle.py``), starting from the highest-likelihood points
of the Nested Sampling chain.

This module depends on JAX, ``ARIMA_ns``, and ``ARIMA_fast``.
All file input/output and plotting are delegated to
``arima_results.py``, which does not have these dependencies. See
that module to load and plot a results file produced by a run
elsewhere.
"""

import os
import warnings

import numpy as np
from scipy.special import logsumexp

from ARIMA_ns import loglikelihood, prior_parameters, ARIMA_Nested_Sampler
from ARIMA import ARIMA_fast
from arima_mle import mle_for_order, samples_to_unconstrained

import arima_results as ar
import pandas as pd
from statsmodels.tsa.stattools import kpss,adfuller

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
    """Approximate the D0 (initial-condition) Occam-factor contribution.

    For each initial-condition parameter ``init_y_i``, the function
    calculates ``log(sigma_post / tau)``, where ``sigma_post`` is the
    posterior standard deviation of that parameter and ``tau`` is
    its prior scale. The supplied ``tau`` must match the prior scale
    used for ``init_y`` in ``prior_parameters()``. For the current
    ``prior_type='normal'`` and ``prior_type='pacf'`` implementations,
    this is ``mu_scale``; callers should therefore pass
    ``self.mu_scale``.

    This is the same family of approximation as the D_KL reported
    in Fig. 8: a posterior much narrower than the prior gives a
    strongly negative contribution, while a posterior comparable
    to the prior width gives a contribution near zero. This is an
    approximate per-parameter diagnostic. It assumes near-Gaussian
    marginal distributions and treats them independently of
    ``phi``, ``theta``, ``sigma``, and ``mu``; it is not an exact
    evidence decomposition.

    Parameters
    ----------
    posterior_samples : object
        Posterior samples containing the ``init_y_i`` parameters.
    order : tuple of int
        ARIMA model order ``(p, d, q)``. The AR order ``p`` determines
        the number of initial-condition parameters.
    tau : float
        Prior scale of the initial-condition parameters.

    Returns
    -------
    tuple
        ``(total, per_param)``, where ``total`` is the sum of the
        contributions and ``per_param`` is a dictionary mapping each
        ``init_y_i`` parameter name to its contribution. For ``p=0``,
        returns ``(0.0, {})`` because there are no D0 parameters to
        penalise.
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
    """Perform grid-based Bayesian comparison of ARIMA(p, d, q) models.

    The class runs Nested Sampling for each model in a specified
    grid and stores the resulting model-comparison quantities as
    instance attributes. Call ``.run()`` once to perform the grid
    search. A previously saved evidence file can instead be loaded
    using ``.load_evidence_file()``.
    """

    def __init__(self, data, max_p, max_q, d, num_live, num_delete, seed,
                 prior_type="normal", mu_mean=0, mu_scale=1, prior_scale=1,
                 inner_steps_factor=6, file_name=None, prior_bounds=None,
                 include_mean_param=True, include_scale_param=True,meas_sigma=None,
                 maximum_likelihood=False, mle_top_k=10, mle_n_random=0, mle_maxiter=500):
        """Initialise the ARIMA model-comparison configuration.

        Parameters
        ----------
        data : array-like
            Time series data to be analysed.
        max_p : int
            Maximum autoregressive order included in the model grid.
        max_q : int
            Maximum moving-average order included in the model grid.
        d : int
            Differencing order of the ARIMA models.
        num_live : int
            Number of live points used by Nested Sampling.
        num_delete : int
            Number of points deleted at each Nested Sampling iteration.
        seed : int
            Random seed.
        prior_type : str, optional
            Type of prior used for the model parameters. The recognised
            branches in the grid search are ``"pacf"`` and ``"uniform"``;
            other values use the ``"normal"`` branch.
        mu_mean : float, optional
            Prior mean of the long-term mean of the data.
        mu_scale : float, optional
            Prior scale of the long-term mean of the data. It also
            serves as ``tau``, the prior scale for the D0 (``init_y``)
            parameters; see ``compute_d0_occam_factor``.
        prior_scale : float, optional
            Prior scale for the autoregressive (``phi``) and
            moving-average (``theta``) coefficients.
        inner_steps_factor : int or float, optional
            Factor passed to ``ARIMA_Nested_Sampler`` to determine its
            inner sampling steps.
        file_name : str or None, optional
            If provided, this file is started fresh for each call to
            ``.run()``. A warning is issued if it already exists, to
            make the overwrite explicit. Results are written as each
            model finishes, so progress is retained if a run is
            interrupted. After the full grid is completed, the file
            is rewritten with the normalised-posterior, BIC, D0, and
            net-prior-volume columns; see ``.run()``.
        prior_bounds : dict or None, optional
            Prior bounds passed to ``ARIMA_Nested_Sampler``. If
            ``None``, an empty dictionary is used.
        include_mean_param : bool, optional
            Whether to include a mean parameter in the BIC parameter
            count.
        include_scale_param : bool, optional
            Whether to include a noise-scale parameter in the BIC
            parameter count. The count is
            ``k = p + q + (1 if mean) + (1 if scale)``. These options
            should match the parameters actually fitted by
            ``ARIMA_Nested_Sampler``; see ``num_arima_params()``.
        meas_sigma : array-like or float or None, optional
            Measurement uncertainties passed to ``ARIMA_Nested_Sampler``
            and the maximum-likelihood refinement.
        maximum_likelihood : bool, optional
            If True, the maximum log-likelihood (and hence BIC) of
            each model is refined using BFGS optimisation
            (``arima_mle.py``), starting from the highest-likelihood
            points in the Nested Sampling chain. The Nested Sampling
            maximum is retained as a lower bound, so the refined value
            cannot be lower. If False, the maximum found by Nested
            Sampling is used. Defaults to False.
        mle_top_k : int, optional
            Number of highest-likelihood Nested Sampling points used
            as BFGS starting points. Used only when
            ``maximum_likelihood`` is True.
        mle_n_random : int, optional
            Number of additional random starting points to draw and
            rank for the optimiser. The default, 0, uses only the
            Nested Sampling points. Used only when
            ``maximum_likelihood`` is True.
        mle_maxiter : int, optional
            Maximum number of BFGS iterations per starting point.
            Used only when ``maximum_likelihood`` is True.
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
        self.maximum_likelihood = maximum_likelihood
        self.mle_top_k = mle_top_k
        self.mle_n_random = mle_n_random
        self.mle_maxiter = mle_maxiter

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
        self.ns_max_loglikelihood = None
        self.mle_gain = None

        if self.file_name:
                    if os.path.exists(self.file_name):
                        warnings.warn(
                            f"'{self.file_name}' already exists and will be overwritten by this run. "
                            "If you meant to keep it, rename/move it (or change file_name) before calling .run().",
                            stacklevel=2,
                        )
                    open(self.file_name, "w").close()  # start from a clean file -- never append to stale/old data

        
        

    # ------------------------------------------------------------------ #
    # maximum-likelihood refinement
    # ------------------------------------------------------------------ #

    def _refine_mle(self, model, order, seed, ns_max_logL):
        """Refine one model's maximum log-likelihood using BFGS.

        Optimisation starts from the highest-likelihood points of the
        model's Nested Sampling chain.

        Parameters
        ----------
        model : ARIMA_Nested_Sampler
            Fitted sampler instance. Its ``posterior_samples`` are
            used to construct starting points.
        order : tuple of int
            ARIMA model order ``(p, d, q)``.
        seed : int
            Random seed used for this model's Nested Sampling run.
            It is also passed to the likelihood so that both
            likelihood evaluations are identical.
        ns_max_logL : float
            Maximum log-likelihood found by the Nested Sampling chain.
            This is used as a lower bound for the optimisation.

        Returns
        -------
        tuple
            ``(mle_logL, gain)``, where ``gain`` is
            ``mle_logL - ns_max_logL`` and is non-negative. If no
            usable starting points can be constructed from the chain
            and no random starts are requested, the Nested Sampling
            maximum is returned with a gain of zero.
        """
        ps = model.posterior_samples
        logL = np.asarray(ps.logL, dtype=float)
        idx = np.argsort(logL)[::-1][:self.mle_top_k]

        starts = samples_to_unconstrained(ps, idx, order, self.prior_type)
        if len(starts) == 0 and self.mle_n_random == 0:
            warnings.warn(f"No usable NS start points for {order}; keeping NS max logL.")
            return ns_max_logL, 0.0

        res = mle_for_order(
            data=self.data, order=order,
            likelihood_type='pacf' if self.prior_type == 'pacf' else 'normal',
            floor_logL=ns_max_logL, seed=seed, meas_sigma=self.meas_sigma,
            n_random=self.mle_n_random, top_k=self.mle_top_k,
            maxiter=self.mle_maxiter, rng_seed=int(seed),
            start_points=starts,
        )
        return res['logL_mle'], res['gain']

    # ------------------------------------------------------------------ #
    # running the grid search
    # ------------------------------------------------------------------ #

    def run(self):
        """Run the ARIMA(p, d, q) grid search and populate instance attributes.

        Each model in the grid is evaluated using Nested Sampling.
        If ``self.maximum_likelihood`` is True, its maximum
        log-likelihood (and hence BIC) is refined using BFGS,
        initialised from the highest-likelihood Nested Sampling
        points. The original Nested Sampling maximum is stored in
        ``self.ns_max_loglikelihood`` and the improvement is stored
        in ``self.mle_gain``.

        If ``self.file_name`` is set, results are written in two
        stages:

        1. A line is written as each model finishes, containing
           ``Order``, ``Seed``, ``Evidence``, ``Error``, ``V``,
           ``MaxLogL``, ``BIC``, ``D0Occam``, ``NetPriorVolume``,
           ``NSMaxLogL``, and ``MLEGain``. This preserves progress if
           the run is interrupted. ``NetPriorVolume`` can be written
           per model because it depends only on that model's ``V``
           and ``D0Occam``, not on the full grid's evidences.
        2. After the entire grid is complete, the file is rewritten
           with a ``Posterior`` column added to every line. This is
           done at the end because the normalised posterior
           probability of each model depends on the evidences of
           every other model in the grid.

        Returns
        -------
        self : ARIMA_model_comparison
            The instance, with the grid results populated in its
            attributes.
        """
        evidences, evidence_err, V, max_loglikelihood, BIC = [], [], [], [], []
        ns_max_loglikelihood, mle_gain = [], []
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

            ns_max_logL = float(np.max(model.posterior_samples.logL))
            max_logL, gain = ns_max_logL, 0.0
            if self.maximum_likelihood:
                max_logL, gain = self._refine_mle(model, order, seed, ns_max_logL)
                print(f"MLE refinement {order}: NS max {ns_max_logL:.5f} -> MLE {max_logL:.5f} (gain {gain:.5f})")

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
            ns_max_loglikelihood.append(ns_max_logL)
            mle_gain.append(gain)
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
                            f"NetPriorVolume={net_this}, NSMaxLogL={ns_max_logL}, "
                            f"MLEGain={gain}\n")

        evidences = np.array(evidences)
        evidence_err = np.array(evidence_err)
        V = np.array(V, dtype=float)
        max_loglikelihood = np.array(max_loglikelihood)
        ns_max_loglikelihood = np.array(ns_max_loglikelihood)
        mle_gain = np.array(mle_gain)
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
        self.ns_max_loglikelihood = ns_max_loglikelihood
        self.mle_gain = mle_gain
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
                            f"NSMaxLogL={ns_max_loglikelihood[i]}, MLEGain={mle_gain[i]}, "
                            f"Posterior={log_posteriors[i]}\n")

        return self

    # ------------------------------------------------------------------ #
    # thin wrappers around arima_results.py -- use those functions directly
    # if you don't need/want a class instance
    # ------------------------------------------------------------------ #

    def load_evidence_file(self, file_name=None, check_normalization=True):
        """Load previously saved model-comparison results into this instance.

        The evidence file is loaded into the instance attributes
        (``self.orders``, ``self.evidences``, and the other
        result attributes) without rerunning the sampler. This is
        equivalent to calling ``arima_results.load_evidence_file()``
        directly and unpacking its returned dictionary.

        Parameters
        ----------
        file_name : str or None, optional
            Evidence-file path. If omitted, ``self.file_name`` is
            used. If neither is set, a ``ValueError`` is raised.
        check_normalization : bool, optional
            Whether to check posterior normalisation when loading
            the results.

        Notes
        -----
        ``self.d0_per_param`` is set to ``None`` because the
        per-parameter D0 details are not saved in the text file.
        Only the per-model total, ``self.d0_occam``, is persisted.
        The per-parameter details are available only on a live
        object immediately after ``.run()``.

        Returns
        -------
        self : ARIMA_model_comparison
            The instance, with the loaded results populated in its
            attributes.
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
        """Retrieve a result quantity and its associated errors.

        Parameters
        ----------
        name : str
            Name of the quantity to retrieve. Supported names are
            ``"log_posteriors"``, ``"evidences"``, ``"BIC"``, ``"V"``,
            ``"max_loglikelihood"``, ``"d0_occam"``, ``"log_V_boost"``,
            and ``"net_prior_volume_effect"``.

        Returns
        -------
        tuple
            ``(values, errors)`` for the requested quantity. The
            errors are ``None`` for quantities without associated
            errors.

        Raises
        ------
        ValueError
            If ``name`` is unknown or the corresponding instance
            attribute has not yet been populated by ``.run()`` or
            ``.load_evidence_file()``.
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
        """Plot a heatmap of one model-comparison quantity over the (p, q) grid.

        Parameters
        ----------
        quantity : str, optional
            Quantity to plot. Supported values are ``"log_posteriors"``,
            ``"BIC"``, ``"V"``, ``"evidences"``,
            ``"max_loglikelihood"``, ``"d0_occam"``,
            ``"log_V_boost"``, and ``"net_prior_volume_effect"``.
            Defaults to ``"log_posteriors"``.
        **kwargs
            Additional keyword arguments passed to
            ``arima_results.plot_evidence_heatmap()``. If not
            supplied, ``invert`` defaults to True for BIC (since
            lower BIC is better), and ``title`` and ``cbar_label``
            default to the selected quantity.

        Returns
        -------
        object
            The result returned by ``arima_results.plot_evidence_heatmap()``.
        """
        data = self._quantity(quantity)
        kwargs.setdefault("invert", quantity == "BIC")  # lower BIC is better
        kwargs.setdefault("title", quantity)
        kwargs.setdefault("cbar_label", quantity)
        return ar.plot_evidence_heatmap(data, self.max_p, max_q=self.max_q,
                                         orders=self.orders, **kwargs)

    def compare(self, quantity1="log_posteriors", quantity2=None, labels=None, **kwargs):
        """Plot two model-comparison quantities side by side.

        Parameters
        ----------
        quantity1 : str, optional
            First quantity to plot. Defaults to ``"log_posteriors"``.
        quantity2 : str or None, optional
            Second quantity to plot. If ``None``, the method instead
            produces the ordinary single heatmap of ``quantity1``,
            equivalent to ``plot_evidence_heatmap()``.
        labels : tuple or list or None, optional
            Labels for the two quantities. If ``None``, the quantity
            names are used.
        **kwargs
            Additional keyword arguments passed to
            ``arima_results.plot_comparison_heatmap()`` or, when
            ``quantity2`` is ``None``, to
            ``plot_evidence_heatmap()``. The ``invert`` option
            defaults to a pair indicating whether each quantity is
            BIC.

        Returns
        -------
        object
            The result returned by the corresponding plotting
            function.

        Examples
        --------
        Compare log posterior probabilities with BIC::

            comp.compare("log_posteriors", "BIC")

        Compare log posterior probabilities with the acceptance rate
        or the net prior-volume effect::

            comp.compare("log_posteriors", "V")
            comp.compare("log_posteriors", "net_prior_volume_effect")
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
        """Quantify the prior-volume and likelihood contributions to a pairwise comparison.

        For two ARIMA model orders, this method separates the raw
        log-posterior difference into the net prior-volume-effect
        difference and the remaining likelihood-only difference.
        The net prior-volume effect comprises the rejection-sampling
        renormalisation minus the D0 Occam penalty.

        Parameters
        ----------
        order_of_interest : tuple of int
            ARIMA order ``(p, d, q)`` whose result is compared with
            the baseline order.
        baseline_order : tuple of int
            Baseline ARIMA order ``(p, d, q)``.

        Returns
        -------
        dict
            Dictionary containing the raw log-posterior difference,
            the differences in the ``log_V_boost``, D0 Occam factor,
            and net prior-volume effect, the likelihood-only
            difference (the raw difference minus the net
            prior-volume difference), and the fraction of the raw
            difference accounted for by the prior-volume term.
            If the raw log-posterior difference is zero, this
            fraction is ``numpy.nan``.

        Examples
        --------
        For example, for a sunspot grid::

            comp.prior_volume_report(
                order_of_interest=(9, 0, 1),
                baseline_order=(0, 0, 1),
            )
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
