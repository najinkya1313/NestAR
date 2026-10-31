import jax
import jax.numpy as jnp
from blackjax.ns.utils import uniform_prior

##Normal prior distributions for the ARIMA model parameters. The code implements a constrained prior distribution to get stationary and invertible ARMA coefficient parameters.

def normal_prior(rng_key,num_live,prior_params,order):
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

 def valid_coeff_block(rng_key,n_needed,k,sign,label):
  if k == 0:
   return jnp.zeros((n_needed,0)),1.0

  def is_valid(x):
   poly = jnp.concatenate([sign*jnp.flip(x),jnp.ones(1)])
   return jnp.all(jnp.abs(jnp.roots(poly,strip_zeros=False))>1)
  valid_mask = jax.jit(jax.vmap(is_valid))

  batch_size = n_needed*1000
  kept = []
  total_drawn = 0
  total_accepted = 0
  while total_accepted < n_needed:
   rng_key,sample_key = jax.random.split(rng_key)
   x = overall_scale*jax.random.normal(sample_key,(batch_size,k))
   x_valid = x[valid_mask(x)]
   total_drawn += batch_size
   total_accepted += len(x_valid)
   kept.append(x_valid)
   
  return jnp.concatenate(kept)[:n_needed],total_accepted/total_drawn

 rng_key,phi_key,theta_key,rest_key = jax.random.split(rng_key,4)
 phi_block,V_ar = valid_coeff_block(phi_key,num_live,p,-1.0,'AR')
 theta_block,V_ma = valid_coeff_block(theta_key,num_live,q,1.0,'MA')
 V = V_ar*V_ma

 sigma_key,mu_key,init_key = jax.random.split(rest_key,3)
 sigma_particles = scale_sigma*jax.random.truncated_normal(sigma_key,1e-5,jnp.inf,shape=(num_live,)) + mean_sigma
 mu_particles = mu_mean + mu_scale*jax.random.normal(mu_key,(num_live,))

 particles = {}
 for i,label in enumerate(phi_names):
  particles[label] = phi_block[:,i]
 for j,label in enumerate(theta_names):
  particles[label] = theta_block[:,j]
 particles['sigma'] = sigma_particles
 particles['mu'] = mu_particles
 if p:
  init_y_particles = init_y_scale*jax.random.normal(init_key,(num_live,p)) + init_y_mean
  for i,label in enumerate(init_y_names):
   particles[label] = init_y_particles[:,i]
   
 return particles,logprior_fn,V

def prior_pacf_uniform(rng_key, num_live, prior_params, order):
    """
    Uniform(-1,1) prior on the PACF-parametrised ARMA coefficients
    (alpha_ar_i, alpha_ma_j).
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
