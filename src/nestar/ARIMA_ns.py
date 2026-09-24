##Python packages

import time

import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from anesthetic import NestedSamples

from .ARIMA import ARIMA_fast
from .priors import normal_prior, prior_pacf_uniform


def pacf_to_arma(alpha):
    """
    Convert partial autocorrelation coefficients to AR coefficients.

    Uses the Durbin--Levinson recursion of Barndorff-Nielsen and
    Schou (1973) to transform a vector of partial autocorrelation
    coefficients into autoregressive coefficients.

    Parameters
    ----------
    alpha : jax.Array
        One-dimensional array of partial autocorrelation coefficients,
        with values in the interval ``(-1, 1)``.

    Returns
    -------
    jax.Array
        One-dimensional array of autoregressive coefficients obtained
        from the supplied partial autocorrelations.

    Notes
    -----
    For parameters in the open interval ``(-1, 1)``, the resulting
    autoregressive coefficients satisfy the stationarity constraint.
    """
    k = alpha.shape[0]
    if k == 0:
        return jnp.array([])
    phi = jnp.zeros(k)
    phi = phi.at[0].set(alpha[0])
    for i in range(1, k):
        prev = phi[:i]
        updated = prev - alpha[i] * jnp.flip(prev)
        phi = phi.at[:i].set(updated)
        phi = phi.at[i].set(alpha[i])
    return phi

#######################################################################################
#--------------------- Likelihood Functions -------------------------------------------
#######################################################################################

def loglikelihood(data, order, seed, meas_sigma=None, custom=False, custom_llk=None):
    """
    Construct the log-likelihood function for an ARIMA model.

    The returned likelihood function evaluates the log-likelihood for
    a given set of ARIMA model parameters. Measurement uncertainties,
    when provided, are combined in quadrature with the free
    process-noise parameter.

    Parameters
    ----------
    data : array-like
        One-dimensional time-series data.
    order : tuple of int
        ARIMA model order ``(p, d, q)``, where ``p`` is the
        autoregressive order, ``d`` is the differencing order, and
        ``q`` is the moving-average order.
    seed : int
        Random seed passed to the ARIMA model evaluation.
    meas_sigma : array-like, optional
        Per-observation measurement uncertainties. If provided, the
        total variance is computed as
        ``sigma_t^2 = meas_sigma_t^2 + sigma^2``, where ``sigma`` is
        the fitted process-noise parameter. If ``None``, a
        homoscedastic likelihood is used.
    custom : bool, optional
        If ``True``, return ``custom_llk`` instead of constructing the
        default Gaussian likelihood. Defaults to ``False``.
    custom_llk : callable, optional
        User-supplied log-likelihood function. Required when
        ``custom=True``.

    Returns
    -------
    callable
        A log-likelihood function accepting a parameter dictionary
        and returning the corresponding log-likelihood.
    """
    p, d, q = order
    data = jnp.asarray(data)
    n = data.shape[0]
    if meas_sigma is None:
        meas_var = jnp.zeros(n)
    else:
        meas_var = jnp.asarray(meas_sigma) ** 2
        if meas_var.shape[0] != n:
            raise ValueError(f"meas_sigma length {meas_var.shape[0]} != data length {n}")

    def llk(params):
        sigma = params['sigma']   # now sigma_proc specifically, not total noise
        mu = params['mu']
        phi = jnp.array([params[i] for i in phi_keys]) if p > 0 else jnp.array([])
        theta = jnp.array([params[j] for j in theta_keys]) if q > 0 else jnp.array([])
        init_y = jnp.array([params[k] for k in init_y_keys]) if p > 0 else jnp.array([])

        if mu.shape != ():
            mu = mu.reshape(())

        y_model = ARIMA_fast(data, order, 0, mu, phi, theta, init_y, seed)

        total_var = meas_var + sigma**2                       # sigma_t^2 = sigma_meas,t^2 + sigma_proc^2
        resid = data - y_model
        return -0.5 * jnp.sum(resid**2 / total_var + jnp.log(2 * jnp.pi * total_var))

    if custom == True and custom_llk is not None:
        return custom_llk
    elif custom == True and custom_llk is None:
        raise ValueError("Custom log-likelihood function must be provided if custom is set to True.")
    else:
        return llk


def pacf_loglikelihood(data:array, order:tuple, seed:int, meas_sigma=None):
    """
    Construct a Gaussian log-likelihood function using PACF parameters.

    The AR and MA coefficients are parameterised through partial
    autocorrelation coefficients and transformed to ARMA coefficients
    using the Durbin--Levinson recursion.

    Parameters
    ----------
    data : array-like
        One-dimensional time-series data.
    order : tuple of int
        ARIMA model order ``(p, d, q)``, where ``p`` is the
        autoregressive order, ``d`` is the differencing order, and
        ``q`` is the moving-average order.
    seed : int
        Random seed passed to the ARIMA model evaluation.
    meas_sigma : array-like, optional
        Per-observation measurement uncertainties. If provided, the
        total variance is computed as

        ``sigma_t^2 = meas_sigma_t^2 + sigma^2``.

        If None, a homoscedastic likelihood is used.

    Returns
    -------
    callable
        A log-likelihood function accepting a dictionary of model
        parameters and returning the corresponding log-likelihood.
    """
    
    p, d, q = order
    ar_keys = [f'alpha_ar_{i+1}' for i in range(p)]
    ma_keys = [f'alpha_ma_{j+1}' for j in range(q)]
    init_y_keys = [f'init_y_{k+1}' for k in range(p)]

    data = jnp.asarray(data)
    n = data.shape[0]
    if meas_sigma is None:
        meas_var = jnp.zeros(n)
    else:
        meas_var = jnp.asarray(meas_sigma) ** 2
        if meas_var.shape[0] != n:
            raise ValueError(f"meas_sigma length {meas_var.shape[0]} != data length {n}")

    def llk(params):
        sigma, mu = params['sigma'], params['mu']
        alpha_ar = jnp.array([params[k] for k in ar_keys]) if p else jnp.array([])
        alpha_ma = jnp.array([params[k] for k in ma_keys]) if q else jnp.array([])
        init_y = jnp.array([params[k] for k in init_y_keys]) if p else jnp.array([])

        phi = pacf_to_arma(alpha_ar) if p else jnp.array([])
        theta = -pacf_to_arma(alpha_ma) if q else jnp.array([])

        if mu.shape != ():
            mu = mu.reshape(())
        y_model = ARIMA_fast(data, order, 0, mu, phi, theta, init_y, seed)

        total_var = meas_var + sigma**2
        resid = data - y_model
        return -0.5 * jnp.sum(resid**2 / total_var + jnp.log(2 * jnp.pi * total_var))

    return llk



#######################################################################################
#--------------------- Prior Helper Function -------------------------------------------
#######################################################################################

def prior_parameters(prior_type:str,order:tuple,coeff_scale,mu_mean,mu_scale,prior_bounds={}):
    """
    Construct the prior-parameter dictionary for an ARIMA model.

    Parameters
    ----------
    prior_type : {"normal", "pacf", "uniform"}
        Type of prior parameterisation.
    order : tuple of int
        ARIMA model order ``(p, d, q)``.
    coeff_scale : float
        Scale of the normal prior distributions for the AR and MA
        coefficients when ``prior_type="normal"``.
    mu_mean : float
        Mean of the prior distribution for the long-term mean ``mu``.
    mu_scale : float
        Scale of the prior distribution for the long-term mean ``mu``.
    prior_bounds : dict, optional
        Parameter bounds used when ``prior_type="uniform"``. Must
        contain the bounds required for the selected ARIMA order.

    Returns
    -------
    dict
        Dictionary containing the prior parameters for the ARIMA model.
        
    """
    p,d,q = order
    prior_params = {}
    if prior_type =="normal":
     for ar in range(p):
        prior_params.update({f'phi_{ar+1}':{'mean':0,'scale':coeff_scale}})
     for ma in range(q):
        prior_params.update({f'theta_{ma+1}':{'mean':0,'scale':coeff_scale}})
     init_y_scale = mu_scale
     init_y_mean = mu_mean
     prior_params.update({'sigma':{'mean':0,'scale':20}})
     prior_params.update({'mu':{'mean':mu_mean,'scale':mu_scale}})
     for in_y in range(p):
         prior_params.update({f'init_y_{in_y+1}':{'mean':init_y_mean,'scale':init_y_scale}})
     


    elif prior_type == "pacf":
        # alpha bounds not actually consumed from here (prior_pacf_uniform
        # hardcodes -1,1) -- kept for the key set / self.columns / bookkeeping.
        for ar in range(p):
            prior_params.update({f'alpha_ar_{ar+1}': (-1.0, 1.0)})
        for ma in range(q):
            prior_params.update({f'alpha_ma_{ma+1}': (-1.0, 1.0)})
        init_y_scale, init_y_mean = mu_scale, mu_mean
        prior_params.update({'sigma':{'mean':0,'scale':20}})
        prior_params.update({'mu':{'mean':mu_mean,'scale':mu_scale}})
        for in_y in range(p):
            prior_params.update({f'init_y_{in_y+1}':{'mean':init_y_mean,'scale':init_y_scale}})

    elif prior_type == "uniform":
        if len(prior_bounds)==0:
            raise ValueError("Missing prior_bounds for uniform prior.")
        for ar in range(p):
            prior_params.update({f'phi_{ar+1}': prior_bounds[f'phi_{ar+1}']})
        for ma in range(q):
            prior_params.update({f'theta_{ma+1}':prior_bounds[f'theta_{ma+1}']})
        prior_params.update({'sigma':prior_bounds['sigma']})
        prior_params.update({'mu':prior_bounds['k']})

    else:
        raise SyntaxError("prior_type should be 'normal', 'pacf', or 'uniform'.")

    return prior_params


#######################################################################################
#------------------------------ Nested Sampler ----------------------------------------
#######################################################################################

class ARIMA_Nested_Sampler:
 """
    Perform Bayesian inference for an ARIMA model using Nested Sampling.

    This class runs a nested-sampling calculation for a specified
    ARIMA(p, d, q) model using BlackJAX. The resulting posterior
    samples and Bayesian evidence are stored as attributes of the
    fitted instance.

    In addition to fitting the model, the class provides methods for
    summarising the posterior, obtaining a fit evaluated at the
    posterior-mean parameters, and performing in-sample and
    out-of-sample forecasting.

    Parameters
    ----------
    data : array-like
        One-dimensional time-series data to be modelled.
    order : tuple of int
        ARIMA model order ``(p, d, q)``.
    mu_mean : float
        Mean of the prior distribution for the long-term mean ``mu``.
    mu_scale : float
        Scale of the prior distribution for the long-term mean ``mu``.
    num_live : int
        Number of live points used by the nested sampler.
    num_delete : int
        Number of live points removed at each nested-sampling
        iteration.
    seed : int
        Random seed used to initialise the sampling calculation.
    inner_steps_factor : int, optional
        Factor controlling the number of inner sampling steps.
        The number of inner steps is given by
        ``inner_steps_factor * ndim``, where ``ndim`` is the number
        of model parameters. Default is 6.
    prior_scale : float, optional
        Scale of the normal priors on the AR and MA coefficients.
        Default is 1.
    prior_type : {"normal", "pacf", "uniform"}, optional
        Prior parameterisation used for the model. Default is
        ``"normal"``.
    meas_sigma : array-like, optional
        Per-observation measurement uncertainties. If provided,
        measurement and process-noise variances are combined in
        quadrature in the likelihood.
    prior_bounds : dict, optional
        Parameter bounds used when ``prior_type="uniform"``.

    Attributes
    ----------
    data : jax.Array
        Input time-series data.
    order : tuple of int
        ARIMA model order ``(p, d, q)``.
    mu_mean : float
        Prior mean for the long-term mean.
    mu_scale : float
        Prior scale for the long-term mean.
    num_live : int
        Number of nested-sampling live points.
    num_delete : int
        Number of points removed per nested-sampling iteration.
    seed : int
        Random seed used for the sampling calculation.
    prior_scale : float
        Scale of the AR and MA coefficient priors.
    prior_type : str
        Prior parameterisation used by the sampler.
    prior_bounds : dict
        Bounds supplied for a uniform prior.
    meas_sigma : array-like or None
        Per-observation measurement uncertainties.
    prior_params : dict
        Prior-parameter specification used to initialise the sampler.
    particles : dict
        Initial live particles sampled from the prior.
    V : float or None
        Prior-volume factor associated with the selected prior.
    posterior_samples : anesthetic.NestedSamples
        Posterior samples produced by the nested-sampling calculation.
    posterior_means : list
        Posterior mean value of each fitted parameter.
    log_evidence : float
        Estimated logarithm of the Bayesian evidence.
    log_evidence_err : float
        Estimated uncertainty in the log Bayesian evidence.
    ns_time : float
        Runtime of the nested-sampling calculation, in seconds.
    forecast_results : object
        Results from the most recent out-of-sample forecast.
    insample_results : object
        Results from the most recent in-sample forecast.
    y_fit : jax.Array
        Posterior-mean fitted time series generated by
        :meth:`mean_fit_plot`.
    """
 def __init__(self,data,order,mu_mean,mu_scale,num_live,num_delete,seed,inner_steps_factor=6,prior_scale=1,
              prior_type="normal",meas_sigma=None,prior_bounds={}):
  
  self.data = jnp.asarray(data)
  self.order = order
  self.mu_mean = mu_mean
  self.mu_scale = mu_scale
  
  self.num_live = num_live
  self.num_delete = num_delete
  self.seed = seed
  self.prior_scale = prior_scale
  p,d,q = self.order
  self.prior_bounds = prior_bounds
  self.prior_type = prior_type
  self.meas_sigma = meas_sigma


  prior_params = prior_parameters(self.prior_type,self.order,self.prior_scale,self.mu_mean,self.mu_scale,self.prior_bounds)
  self.prior_params = prior_params
  if prior_type == 'pacf':
    self.log_likelihood = pacf_loglikelihood(self.data, self.order, self.seed,meas_sigma=self.meas_sigma)
  else:
    self.log_likelihood = loglikelihood(self.data, self.order, self.seed,meas_sigma=self.meas_sigma)
 
  
    
  print(f"Running Nested Sampling for fitting ARIMA {self.order} model...")
  num_dims = len(self.prior_params)
  num_inner_steps = num_dims * inner_steps_factor
  p,d,q = self.order
 
    
  rng_key = jax.random.PRNGKey(self.seed)
  rng_key,prior_key = jax.random.split(rng_key)
  if prior_type=='normal':
   particles,logprior_fn,V = normal_prior(prior_key,self.num_live,self.prior_params,self.order)
   self.V = V

  elif prior_type == 'pacf':
    particles, logprior_fn = prior_pacf_uniform(prior_key, self.num_live, self.prior_params, self.order)
    self.V = 1.0   

  elif prior_type=='uniform':
   particles,logprior_fn = blackjax.ns.utils.uniform_prior(prior_key,self.num_live,self.prior_params)
   self.V = None

  else:
    raise SyntaxError(f"Invalid prior_type '{prior_type}'. prior_type should be 'normal', 'pacf' or 'uniform'")
  self.particles = particles
  
  ##Nested Sampler
  nested_sampler = blackjax.nss(logprior_fn=logprior_fn,loglikelihood_fn = self.log_likelihood,num_delete=self.num_delete,num_inner_steps=num_inner_steps)
  init_fn = jax.jit(nested_sampler.init)
  step_fn = jax.jit(nested_sampler.step)
  ns_start = time.time()
  live = init_fn(particles)
  dead = []
    
  with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not live.logZ_live - live.logZ < jnp.log(1e-3):  # Convergence criterion
      rng_key, subkey = jax.random.split(rng_key, 2)
      live, dead_info = step_fn(subkey, live)
      dead.append(dead_info)
      pbar.update(self.num_delete)
      
     
    
  dead = blackjax.ns.utils.finalise(live,dead)
  ns_time = time.time() - ns_start
  self.ns_time = ns_time
  print(f"Finished Nested Sampling with a total runtime of : {ns_time:.2f} seconds")

 
  ##Processing results
  columns = [i for i in self.prior_params.keys()]
  self.columns = columns
  if prior_type == 'pacf':
    labels = ([fr'$\alpha^{{AR}}_{ph+1}$' for ph in range(p)]
              + [fr'$\alpha^{{MA}}_{th+1}$' for th in range(q)]
              + [r'$\sigma$', r'$\mu$']
              + [fr'$y_{in_y+1}$' for in_y in range(p)])
  else:
    labels = ([fr'$\phi_{ph+1}$' for ph in range(p)]
              + [fr'$\theta_{th+1}$' for th in range(q)]
              + [r'$\sigma$', r'$\mu$']
              + [fr'$y_{in_y+1}$' for in_y in range(p)])
    
  data = jnp.vstack([dead.particles[key] for key in columns]).T

  posterior_samples = NestedSamples(
  data,
  logL=dead.loglikelihood,
  logL_birth=dead.loglikelihood_birth,
  columns=columns,
  labels=labels,
  logzero=jnp.nan,
  )
  self.posterior_samples = posterior_samples
  Z = self.posterior_samples.logZ()
  posterior_means = []
  for key in self.prior_params.keys():
     means = self.posterior_samples[key].mean()
     posterior_means.append(means)
  self.posterior_means = posterior_means
  self.log_evidence = Z
  self.log_evidence_err = self.posterior_samples.logZ(100).std()


    

#Print results:
 def summary(self):
    """
    Print a summary of the nested-sampling results.

    The summary includes the nested-sampling runtime, posterior mean
    of each fitted parameter, and the estimated logarithm of the
    Bayesian evidence. A two-dimensional posterior plot is also
    generated.

    Returns
    -------
    None
    """
    print("||NESTED SAMPLING SUMMARY RESULTS||")
    print("----------------------------------------------------")
    print(f"Nested sampling runtime: {self.ns_time:.2f} seconds")
    print("----------------------------------------------------")
    for index,key in enumerate(self.prior_params.keys()):
     print(f"Posterior mean for {key} : {self.posterior_means[index]}")
    print("---------------------------------------------------")
    
    print(f"Log Evidence: {self.posterior_samples.logZ():.2f} ± {self.posterior_samples.logZ(100).std():.2f}")
    print("-------x----------------x-------------x------------")
    
    
     # Create posterior corner plot with true values marked
    kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
    axes = self.posterior_samples.plot_2d(self.columns, kinds=kinds, label='Posterior')
    plt.suptitle("Posterior Distributions")
    
 def get_mean_forecasts(self):
     """
    Evaluate the ARIMA model at the posterior-mean parameters.

    Returns
    -------
    jax.Array
        Fitted time series evaluated using the posterior mean of
        each model parameter.
    """
     p,d,q = self.order
     y_fit = ARIMA_fast(self.data,self.order,self.posterior_means[p+q],self.posterior_means[p+q+1],self.posterior_means[0:p],self.posterior_means[p:p+q],self.posterior_means[p+q+1:2*p+q+1],self.seed)
     return y_fit
     
  
 def mean_fit_plot(self,compare=None):
    """
    Plot the ARIMA fit obtained from the posterior-mean parameters.

    Parameters
    ----------
    compare : bool or None, optional
        If True, plot both the observed data and the fitted ARIMA
        model. If None, plot only the fitted model. Other values
        raise a ``SyntaxError``.

    Returns
    -------
    None
        The fitted time series is stored in the ``y_fit`` attribute
        and the plot is displayed using Matplotlib.
    """
   p,d,q = self.order
   y_fit = ARIMA_fast(self.data,self.order,self.posterior_means[p+q],self.posterior_means[p+q+1],self.posterior_means[0:p],self.posterior_means[p:p+q],self.posterior_means[p+q+1:2*p+q+1],self.seed)
   self.y_fit = y_fit
   if compare is not None:
    if compare==True:
      plt.plot(self.data,label='Data')
      plt.plot(y_fit,label=f'ARIMA {self.order} model')
     
    elif type(compare)!= bool:
      raise SyntaxError(f"Invalid value {compare} for compare argument. compare should be == True, False, or None")
    
   else:
    plt.plot(y_fit,label=f'ARIMA {self.order} model')
    plt.legend()
    plt.xlabel('Time-step')
    plt.ylabel('Value')
    plt.show()

 def outsample_forecast(self, overall_time, overall_data, num_forecast, upper_index=None,
                         n_samples=1000, seed=None,
                         plot_nested=True, plot_climatology=True, plot_persistence=True,
                         ax=None, **kwargs):
     """
    Generate posterior-based out-of-sample forecasts.

    The posterior samples from the nested-sampling calculation are
    passed to ``PosteriorResults`` to generate forecasts. The results
    are stored in the ``forecast_results`` attribute.

    Parameters
    ----------
    overall_time : array-like
        Time values corresponding to ``overall_data``.
    overall_data : array-like
        Complete time series containing the training data and, when
        applicable, the test data.
    num_forecast : int
        Number of future time steps to forecast.
    upper_index : int or None, optional
        Index separating the training and test portions of
        ``overall_data``. If None, ``overall_data`` is treated as the
        training data and no test-data scoring is performed.
    n_samples : int, optional
        Number of posterior samples used for forecasting. Default is
        1000.
    seed : int or None, optional
        Random seed used for forecasting. If None, the seed from the
        nested-sampling run is used.
    plot_nested : bool, optional
        Whether to plot the nested-sampling posterior forecast.
        Default is True.
    plot_climatology : bool, optional
        Whether to include a climatology comparison. Default is True.
    plot_persistence : bool, optional
        Whether to include a persistence comparison. Default is True.
    ax : matplotlib.axes.Axes or None, optional
        Existing Matplotlib axes on which to draw the forecast.
    **kwargs
        Additional keyword arguments passed to the underlying
        forecasting method.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the forecast.
    results : object
        Forecast results returned by ``PosteriorResults``.

    Notes
    -----
    The forecast results are also stored in the
    ``forecast_results`` attribute.
    """
     from .arima_results import PosteriorResults
     forecaster = PosteriorResults(self.posterior_samples, self.data, order=self.order, prior_type=self.prior_type,
                                    seed=seed if seed is not None else self.seed)
     fig, results = forecaster.outsample_forecast(
         overall_time, overall_data, num_forecast, upper_index=upper_index,
         n_samples=n_samples, seed=seed if seed is not None else self.seed,
         plot_nested=plot_nested, plot_climatology=plot_climatology,
         plot_persistence=plot_persistence, ax=ax, **kwargs)
     self.forecast_results = results
     return fig, results

 def insample_forecast(self, training_time, n_samples=1000, seed=None, ax=None, **kwargs):
     """
    Generate posterior-based in-sample forecasts.

    The posterior samples from the nested-sampling calculation are
    passed to ``PosteriorResults`` to generate an in-sample fit and
    residual analysis. The results are stored in the
    ``insample_results`` attribute.

    Parameters
    ----------
    training_time : array-like
        Time values corresponding to the training data.
    n_samples : int, optional
        Number of posterior samples used for the forecast. Default is
        1000.
    seed : int or None, optional
        Random seed used for forecasting. If None, the seed from the
        nested-sampling run is used.
    ax : matplotlib.axes.Axes or None, optional
        Existing Matplotlib axes on which to draw the forecast.
    **kwargs
        Additional keyword arguments passed to the underlying
        forecasting method.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the in-sample fit and residual analysis.
    results : object
        Forecast results returned by ``PosteriorResults``.

    Notes
    -----
    The forecast results are also stored in the
    ``insample_results`` attribute.
    """
     from .arima_results import PosteriorResults
     forecaster = PosteriorResults(self.posterior_samples, self.data, order=self.order, prior_type=self.prior_type,
                                    seed=seed if seed is not None else self.seed)
     fig, results = forecaster.insample_forecast(
         training_time, n_samples=n_samples, seed=seed if seed is not None else self.seed,
         ax=ax, **kwargs)
     self.insample_results = results
     return fig, results
