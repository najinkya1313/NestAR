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