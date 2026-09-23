import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1,))        
def ARIMA_fast(data,order,sigma,mu, phi,theta,init_y,seed):
    """
    Vectorised non-seasonal ARIMA(p,d,q) for JAX/XLA.
    """
    key = jax.random.PRNGKey(seed)
    key,arima_key = jax.random.split(key)
    p, d, q = order
    x_dtype = jnp.asarray(data).dtype           
    data    = jnp.asarray(data, dtype=jnp.float32)  

    # 1. Differencing -------------------------------------------------------------
    diff = jnp.diff(data, n=d) if d else data

    # 2. Parameters / intercept ---------------------------------------------------
    phi   = jnp.pad(jnp.asarray(phi,   dtype=diff.dtype), (0, p - len(phi)))
    theta = jnp.pad(jnp.asarray(theta, dtype=diff.dtype), (0, q - len(theta)))
    k = mu * (1 - jnp.sum(phi))   # mu = unconditional mean
    

    # 3. Initial state ------------------------------------------------------------
    past_y = jnp.array(init_y) if p else jnp.empty((0,), diff.dtype)
    past_e = jnp.zeros(q) if q else jnp.empty((0,), diff.dtype)

    # 4. One scan step ------------------------------------------------------------
    def one_step(carry, x):
        past_y, past_e = carry

        y_hat = k
        if p:
            y_hat += (phi * past_y).sum()
        if q:
            y_hat += (theta * past_e).sum()

        err = x - y_hat

        if p:
            past_y = jnp.concatenate([jnp.array([x], diff.dtype), past_y[:-1]])
        if q:
            past_e = jnp.concatenate([jnp.array([err], diff.dtype), past_e[:-1]])

        return (past_y, past_e), y_hat

    # 5. Run the recurrence -------------------------------------------------------
    (_, _), y_hat_seq = jax.lax.scan(one_step, (past_y, past_e), diff)

    # 6. Undo differencing --------------------------------------------------------
    if d:
        recovered = jnp.concatenate(
            [data[:d], jnp.cumsum(y_hat_seq) + data[d-1]]
        )
    else:
        recovered = y_hat_seq

    # 7. Optional noise -----------------------------------------------------------
    recovered = jax.lax.cond(
        sigma == 0,
        lambda r: r,
        lambda r: r + sigma * jax.random.normal(arima_key, r.shape, dtype=r.dtype),
        recovered,
    )

    return recovered

def ARIMA_forecast(data, order, sigma, mu, phi, theta, forecast_num, init_y, seed):
    p, d, q = order
    y_model = ARIMA_fast(data, order, 0, mu, phi, theta, init_y, seed)  # sigma=0 for clean fit
    
    phi_coeffs = jnp.array(phi) if p > 0 else jnp.empty(0)
    theta_coeffs = jnp.array(theta) if q > 0 else jnp.empty(0)
    k = mu * (1 - jnp.sum(phi_coeffs))
    
    rng_key = jax.random.PRNGKey(seed)
    error_keys = jax.random.split(rng_key, forecast_num)
    
    # work on differenced scale
    diff_data = jnp.diff(data, n=d) if d else data
    diff_model = jnp.diff(y_model, n=d) if d else y_model
    
    epsilon_lagged = (diff_data[-q:] - diff_model[-q:]) if q > 0 else jnp.empty(0)
    history = diff_data  # rolling window for AR terms
    
    forecasted_diff = []
    for key in error_keys:
        y_phis = (phi_coeffs * jnp.flip(history[-p:])).sum() if p > 0 else 0.0
        y_thetas = (theta_coeffs * jnp.flip(epsilon_lagged)).sum() if q > 0 else 0.0
        epsilon_t = sigma * jax.random.normal(key)
        y_forecast = k + y_phis + y_thetas + epsilon_t
        
        forecasted_diff.append(y_forecast)
        history = jnp.concatenate([history, jnp.array([y_forecast])])
        if q > 0:
            epsilon_lagged = jnp.concatenate([jnp.array([epsilon_t]), epsilon_lagged[:-1]])
    
    forecast_array = jnp.array(forecasted_diff)
    
    # undo differencing
    if d > 0:
        for _ in range(d):
            if d > 1:
              initial = data[-d:-(d-1)]
            else:
              initial = data[-1:]
            forecast_array = jnp.cumsum(jnp.concatenate([initial, forecast_array]))[1:]
    
    return forecast_array
