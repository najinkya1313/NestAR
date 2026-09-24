"""Prior distributions for nested-sampling fits of ARIMA models.

Provides two priors over the parameters of an ARMA(p, q) model with unknown
noise scale ``sigma``, mean ``mu`` and initial values ``init_y``:

- ``normal_prior``: independent Gaussian priors on the AR (``phi``) and MA
  (``theta``) coefficients, restricted by rejection sampling to the region in
  which the model is stationary and invertible.
- ``prior_pacf_uniform``: Uniform(-1, 1) priors on the partial-autocorrelation
  (PACF) parametrisation of the coefficients, which is stationary and
  invertible by construction, so no rejection sampling is needed.

Both return a dictionary of ``num_live`` initial particles together with a
function that evaluates the log prior, for use with the ``blackjax`` nested
sampler.
"""

import jax
import jax.numpy as jnp
from blackjax.ns.utils import uniform_prior

##Normal prior distributions for the ARIMA model parameters. The code implements a constrained prior distribution to get stationary and invertible ARMA coefficient parameters.

def normal_prior(rng_key,num_live,prior_params,order):
 """Constrained Gaussian prior for the parameters of an ARMA model.

 Draws ``num_live`` particles from independent Gaussian priors on the AR
 coefficients ``phi_i`` and MA coefficients ``theta_j``, a truncated Gaussian
 prior on the noise scale ``sigma``, and Gaussian priors on the mean ``mu`` and
 the initial values ``init_y_i``. The particles are restricted, by rejection
 sampling, to the region in which the ARMA process is stationary and
 invertible, i.e. where all roots of the AR polynomial
 ``1 - phi_1 z - ... - phi_p z^p`` and of the MA polynomial
 ``1 + theta_1 z + ... + theta_q z^q`` lie outside the unit circle.

 Parameters
 ----------
 rng_key : jax.Array
     JAX PRNG key used to draw the particles.
 num_live : int
     Number of particles to return.
 prior_params : dict
     Mapping from parameter name to a dict with keys ``'mean'`` and
     ``'scale'`` (the mean and standard deviation of that parameter's
     Gaussian). It must contain entries for ``'phi_1'`` ... ``'phi_p'``,
     ``'theta_1'`` ... ``'theta_q'``, ``'sigma'``, ``'mu'`` and, if ``p > 0``,
     ``'init_y_1'`` ... ``'init_y_p'``.
 order : tuple of int
     ARIMA order ``(p, d, q)``. Only ``p`` and ``q`` are used.

 Returns
 -------
 particles : dict of {str: jax.Array}
     The ``num_live`` accepted particles. Each entry has shape
     ``(num_live,)``, with keys ``phi_i``, ``theta_j``, ``sigma``, ``mu`` and
     ``init_y_i``.
 logprior_fn : callable
     Function ``logprior_fn(params)`` that takes a dict of parameter values
     (a single particle) and returns the scalar log prior density, which is
     ``-inf`` outside the stationary and invertible region.
 V : float
     Acceptance rate of the rejection sampling: the fraction of all drawn
     particles that lie in the stationary and invertible region.

 Notes
 -----
 Particles are drawn in batches of ``1000 * num_live``. Further batches are
 drawn, printing the running acceptance rate, until at least ``num_live``
 particles have been accepted, and the first ``num_live`` are returned.

 Inside the valid region, ``logprior_fn`` is the product of the unrestricted
 densities; it is not renormalised over that region. The acceptance rate
 ``V`` is an estimate of the prior mass of the region and can be used to
 correct for this.

 When drawing particles, all ARMA coefficients are sampled as zero-mean
 Gaussians with the scale of the first coefficient, and all ``init_y_i`` with
 the mean and scale of ``init_y_1``, whereas ``logprior_fn`` uses each
 parameter's own mean and scale. For the particles to follow ``logprior_fn``,
 the ARMA coefficients should therefore share a zero mean and a common scale,
 and the ``init_y_i`` a common mean and scale.

 The lower truncation bound ``1e-5`` of the ``sigma`` prior is applied in
 standardised units (as in ``scipy.stats.truncnorm``), so the effective lower
 bound on ``sigma`` is approximately its prior mean rather than zero.
 """
 p,d,q = order
 phi_names = [f"phi_{i+1}" for i in range(p)]
 theta_names = [f"theta_{j+1}" for j in range(q)]
 init_y_names = [f"init_y_{i+1}" for i in range(p)]
 
 prior_params_modsigma = {name: prior_params[name] for name in phi_names + theta_names}
 init_params = {init_name:prior_params[init_name] for init_name in init_y_names}
 prior_params_sigma = prior_params['sigma']
 mean_sigma = prior_params_sigma['mean']
 scale_sigma = prior_params_sigma['scale']
 prior_params_mu = prior_params['mu']
 mu_mean = prior_params_mu['mean']
 mu_scale = prior_params_mu['scale']
 overall_scale = list(prior_params_modsigma.values())[0]['scale']
 if p:
  init_y_1 = prior_params['init_y_1']
  init_y_scale = init_y_1['scale']
  init_y_mean = init_y_1['mean']

 ##-------------------------------------------------Logprior function–------------------------------------------
 def logprior_fn(params):
  logprior = 0.0
  parameters_mod_sigma = jnp.array([params[key] for key in prior_params_modsigma.keys()])
  ar_parameters = jnp.flip(parameters_mod_sigma[0:p])
  ma_parameters = jnp.flip(parameters_mod_sigma[p:p+q])
  const = jnp.ones(1)
  poly_ar = jnp.concatenate([-ar_parameters,const])
  poly_ma = jnp.concatenate([ma_parameters,const])
  roots_phi = jnp.roots(poly_ar,strip_zeros=False)
  roots_ma = jnp.roots(poly_ma,strip_zeros=False)
 
  for parameter, norm_params in prior_params_modsigma.items():
   x = params[parameter]
   mean = norm_params['mean']
   scale = norm_params['scale']
   logprior += jax.scipy.stats.norm.logpdf(x, mean, scale)
  valid = jnp.all(jnp.abs(roots_phi) > 1) & jnp.all(jnp.abs(roots_ma) > 1)
  output = jnp.where(valid, logprior, -jnp.inf)
  
##For sigma and k:
  x_sig = params["sigma"]
  mu = params['mu']
  logprior_sigma = jax.scipy.stats.truncnorm.logpdf(x_sig,1e-5,jnp.inf,mean_sigma,scale_sigma)
  logprior_mu = jax.scipy.stats.norm.logpdf(mu,mu_mean,mu_scale)
  logprior = output + logprior_sigma + logprior_mu

  for init_param,init_norm_param in init_params.items():
   x = params[init_param]
   mean = init_norm_param['mean']
   scale = init_norm_param['scale']
   logprior+= jax.scipy.stats.norm.logpdf(x,mean,scale)
  return logprior
  

  

##---------------------------------------------------Particle sampler-----------------------------------------------:
 @jax.jit
 def prior_sample(rng_key):
  
  coeff_keys = jax.random.split(rng_key, p+q) ##these are random keys for the ARMA coeffs
  param_labels = [label for label in prior_params_modsigma.keys()]
  phi_labels = param_labels[0:p]
  theta_labels = param_labels[p:p+q]
  params = {}
  particles_all = overall_scale*jnp.array([jax.random.normal(rng_key) for rng_key in coeff_keys]) ##coeff particles
 

  rng_key,sigma_key = jax.random.split(rng_key) ##random keys for sigma particles
  sigma_particle = scale_sigma*(jax.random.truncated_normal(sigma_key,1e-5,jnp.inf)) + mean_sigma

  rng_key,mu_key = jax.random.split(rng_key) ##random keys for mu particles
  mu_particle = mu_mean + mu_scale*(jax.random.normal(mu_key))

  init_keys = jax.random.split(rng_key,p)
  if p:
   init_y_particles = init_y_scale * jnp.array([jax.random.normal(init_y_key) for init_y_key in init_keys[0:p]]) + init_y_mean
 
  

  
 
 #Roots calculation
  phi_particles = particles_all[0:p]
  theta_particles = particles_all[p:p+q]
  theta_particles_flipped = jnp.flip(theta_particles)
  phi_particles_flipped = jnp.flip(phi_particles)
  const = jnp.ones(1)
  phi_poly = jnp.concatenate([-phi_particles_flipped,const])
  theta_poly = jnp.concatenate([theta_particles_flipped,const])
  roots_phi = jnp.roots(phi_poly,strip_zeros=False)
  roots_theta = jnp.roots(theta_poly,strip_zeros=False)
  roots = jnp.concatenate([roots_phi,roots_theta])

  def valid_point(roots):
    for phi_label,phi_particle in zip(phi_labels,phi_particles):
      params.update({phi_label:phi_particle})
    for theta_label,theta_particle in zip(theta_labels,theta_particles):
      params.update({theta_label:theta_particle})
    params.update({'sigma':sigma_particle})
    params.update({'mu':mu_particle})
    if p:
     for init_y_label,init_y_particle in zip(init_y_names,init_y_particles):
       params.update({init_y_label:init_y_particle})
   
    return params
  
  def invalid_point(roots):
    for phi_label,phi_particle in zip(phi_labels,phi_particles):
      params.update({phi_label:0.})
    for theta_label,theta_particle in zip(theta_labels,theta_particles):
      params.update({theta_label:0.})
    params.update({'sigma':0.})
    params.update({'mu':0.})
    if p:
     for init_y_label,init_y_particle in zip(init_y_names,init_y_particles):
       params.update({init_y_label:0.})

    return params
  
  filtered_params = jax.lax.cond(jnp.all(abs(roots)>1),valid_point,invalid_point,roots)
  initlogprior = logprior_fn(filtered_params)
  
  return filtered_params,initlogprior
 ##--------------------------------------------------------------------------------------------------------

 
 ##--------------------------------------Filter to only accept valid particles-------------------------------
 def particles_filter(unfiltered_particles,initlogprior):
   
   
   mask = initlogprior != -jnp.inf
   
   for key,vals in unfiltered_particles.items():
     unfiltered_particles.update({key:vals[mask]})
   
   return unfiltered_particles

 batch_size = num_live*1000
 particle_keys = jax.random.split(rng_key,batch_size)
 unfiltered_particles,unfilteredlogprior = jax.vmap(prior_sample)(particle_keys)
 particles = particles_filter(unfiltered_particles,unfilteredlogprior)
 total_drawn = batch_size
 total_accepted = len(particles['sigma'])
 V = total_accepted/total_drawn
 
 ##------------------While loop to keep drawing samples until num_live reached --------------------------------
 while len(particles['sigma'])<num_live:
   rng_key,sample_key = jax.random.split(rng_key)
   sample_particle_keys = jax.random.split(sample_key,batch_size)
   new_particles,newlogprior = jax.vmap(prior_sample)(sample_particle_keys)
   new_particles_filtered = particles_filter(new_particles,newlogprior)
   total_drawn += batch_size
   total_accepted += len(new_particles_filtered['sigma'])
   V = total_accepted/total_drawn
   print(f"Acceptance rate: {V:.4f}, Total drawn: {total_drawn}")
   for key,vals in new_particles_filtered.items():
     new_arr = jnp.concatenate([particles[key],vals])
     particles.update({key:new_arr})
   
 particles = {label:value[:num_live] for label,value in particles.items()}
   
 return particles,logprior_fn,V

def prior_pacf_uniform(rng_key, num_live, prior_params, order):
    """Uniform prior on the PACF-parametrised ARMA coefficients.

    The AR and MA coefficients are parametrised by partial autocorrelations
    ``alpha_ar_i`` (``i = 1, ..., p``) and ``alpha_ma_j`` (``j = 1, ..., q``),
    each with a Uniform(-1, 1) prior. Every point in this box corresponds to a
    stationary and invertible ARMA model, so no rejection sampling is needed. The
    noise scale ``sigma``, the mean ``mu`` and the initial values ``init_y_i``
    have the same priors as in ``normal_prior``.

    Parameters
    ----------
    rng_key : jax.Array
        JAX PRNG key used to draw the particles.
    num_live : int
        Number of particles to return.
    prior_params : dict
        Mapping from parameter name to a dict with keys ``'mean'`` and
        ``'scale'``. Only the entries ``'sigma'``, ``'mu'`` and, if ``p > 0``,
        ``'init_y_1'`` are read; all ``init_y_i`` share the mean and scale of
        ``init_y_1``. Entries for the ARMA coefficients are not used, since those
        have the uniform prior.
    order : tuple of int
        ARIMA order ``(p, d, q)``. Only ``p`` and ``q`` are used.

    Returns
    -------
    particles : dict of {str: jax.Array}
        The ``num_live`` particles. Each entry has shape ``(num_live,)``, with
        keys ``alpha_ar_i``, ``alpha_ma_j``, ``sigma``, ``mu`` and ``init_y_i``.
    logprior_fn : callable
        Function ``logprior_fn(params)`` that takes a dict of parameter values
        (a single particle) and returns the scalar log prior density.

    Notes
    -----
    Unlike ``normal_prior``, this returns only ``(particles, logprior_fn)``, with
    no acceptance rate, because there is no rejection sampling.

    The coefficient particles and their log prior come from
    ``blackjax.ns.utils.uniform_prior``.

    The lower truncation bound ``1e-5`` of the ``sigma`` prior is applied in
    standardised units (as in ``scipy.stats.truncnorm``), so the effective lower
    bound on ``sigma`` is approximately its prior mean rather than zero.
    """
    p, d, q = order


    alpha_bounds = {}
    for i in range(p):
        alpha_bounds[f'alpha_ar_{i+1}'] = (-1.0, 1.0)
    for j in range(q):
        alpha_bounds[f'alpha_ma_{j+1}'] = (-1.0, 1.0)

    rng_key, alpha_key = jax.random.split(rng_key)
    alpha_particles, alpha_logprior_fn = uniform_prior(alpha_key, num_live, alpha_bounds)
    

    # ---- sigma, mu, D0: same as normal_prior.py ----
    prior_params_sigma = prior_params['sigma']
    mean_sigma, scale_sigma = prior_params_sigma['mean'], prior_params_sigma['scale']
    prior_params_mu = prior_params['mu']
    mu_mean, mu_scale = prior_params_mu['mean'], prior_params_mu['scale']
    init_y_names = [f"init_y_{i+1}" for i in range(p)]
    if p:
        init_y_1 = prior_params['init_y_1']
        init_y_mean, init_y_scale = init_y_1['mean'], init_y_1['scale']

    def rest_logprior_fn(params):
        logprior = jax.scipy.stats.truncnorm.logpdf(
            params['sigma'], 1e-5, jnp.inf, mean_sigma, scale_sigma)
        logprior += jax.scipy.stats.norm.logpdf(params['mu'], mu_mean, mu_scale)
        for name in init_y_names:
            logprior += jax.scipy.stats.norm.logpdf(params[name], init_y_mean, init_y_scale)
        return logprior

    def rest_prior_sample(rng_key):
        params = {}
        rng_key, sig_key, mu_key, init_key = jax.random.split(rng_key, 4)
        params['sigma'] = scale_sigma * jax.random.truncated_normal(sig_key, 1e-5, jnp.inf) + mean_sigma
        params['mu'] = mu_mean + mu_scale * jax.random.normal(mu_key)
        if p:
            init_keys = jax.random.split(init_key, p)
            init_y_vals = init_y_scale * jnp.array(
                [jax.random.normal(k) for k in init_keys]) + init_y_mean
            for name, val in zip(init_y_names, init_y_vals):
                params[name] = val
        return params

    rng_key, rest_key = jax.random.split(rng_key)
    rest_keys = jax.random.split(rest_key, num_live)
    rest_particles = jax.vmap(rest_prior_sample)(rest_keys)

    # ---- merge alpha + rest into one particle dict / one logprior_fn ----
    particles = {**alpha_particles, **rest_particles}

    def logprior_fn(params):
        return alpha_logprior_fn(params) + rest_logprior_fn(params)

    return particles, logprior_fn
