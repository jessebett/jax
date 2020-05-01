"""
Learning exp.
"""

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.experimental.jet import jet
from jax.experimental import optimizers

from jax.config import config
config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.pyplot as plt


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  z_t = jnp.concatenate((jnp.array([z]), jnp.array([t])))

  def g(z_t):
    """
    Closure to expand z.
    """
    z, t = z_t
    dz = jnp.array([f(z, t)])
    dt = jnp.array([1.])
    dz_t = jnp.concatenate((dz, dt))
    return dz_t

  (y0, [y1h]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
  (y0, [y1, y2h]) = jet(g, (z_t, ), ((y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jet(g, (z_t, ), ((y0, y1, y2h), ))
  # (y0, [y1, y2, y3, y4h]) = jet(g, (z_t, ), ((y0, y1, y2, y3h), ))

  return y0[0], [y1[0], y2[0]]
  # return y0[0], [y1[0]]


def _int_pow(x, r):
    """
    Raise x^r where r is an integer by repeated multiplication.
    Avoids numerical instability when x = 0, and r is treated as a float.
    """
    if r == 0:
        return 1
    cache = _int_pow(x, r // 2)
    cache_sq = cache * cache
    if r % 2 == 0:
        return cache_sq
    else:
        return cache_sq * x


def true_dyn(z, t):
    """
    True dynamics to learn.
    """
    return true_degree * _int_pow(t, true_degree - 1)


def dyn(z, t, params):
    """
    Learning polynomial dynamics.
    """
    t_pows = jnp.array([(i + 1) * _int_pow(t, i) for i in range(dyn_degree)])
    return jnp.dot(t_pows, params)


def aug_dyn(z_r, t, params):
    """
    Dynamics augmented with regularization.
    """
    z, r = z_r
    dz = dyn(z, t, params)

    y0, yn = sol_recursive(lambda _z, _t: dyn(_z, _t, params), z, t)
    dr = jnp.sum(lax.square(yn[-1]))

    return dz, dr


true_degree = 2
dyn_degree = 4  # degree of the predicted solution, x-1 degree of dynamics
rng = jax.random.PRNGKey(0)
params = jax.random.normal(rng, (dyn_degree, )) * 0.001
t_endpoints = 0., 1.
ts = jnp.linspace(*t_endpoints, num=4)


@jax.jit
def loss(params):
    """
    Loss for training.
    """
    true_soln = odeint(true_dyn, 0., ts)[0]
    pred_soln, rs = odeint(aug_dyn, (0., 0.), ts, params)[0]
    return jnp.mean(lax.square(true_soln - pred_soln)) + lam * jnp.mean(rs)


@jax.jit
def sep_losses(params):
    """
    Separate out different losses.
    """
    true_soln = odeint(true_dyn, 0., ts)[0]
    pred_soln, rs = odeint(aug_dyn, (0., 0.), ts, params)[0]
    diff = jnp.mean(lax.square(true_soln - pred_soln))
    reg = jnp.mean(rs)
    return diff + lam * reg, diff, reg


grad_fn = jax.grad(loss)
lam = 1
lr = 1e-2
opt_init, opt_update, get_params = optimizers.momentum(step_size=lr, mass=0.9)
opt_state = opt_init(params)


@jax.jit
def update(itr, opt_state):
    """
    Update the parameters using SGD.
    """
    return opt_update(itr, grad_fn(get_params(opt_state)), opt_state)


for itr in range(10000):
    opt_state = update(itr, opt_state)
    if itr % 1000 == 0:
        total_loss_, loss_, reg_ = sep_losses(get_params(opt_state))
        print(total_loss_, loss_, reg_)

params = get_params(opt_state)
print(params)


fig, ax = plt.subplots()

plot_ts = jnp.linspace(*t_endpoints, num=1000)
true_soln = odeint(true_dyn, 0., plot_ts)[0]
pred_soln, rs = odeint(aug_dyn, (0., 0.), plot_ts, params)[0]
learned_soln = odeint(true_dyn, 0., ts)[0]

nfe = odeint(dyn, 0., plot_ts, params)[1]
print(nfe)


def analytic_pred(t):
    """
    Analytically integrate solution.
    """
    t_pows = jnp.array([_int_pow(t, i) for i in range(1, dyn_degree + 1)])
    return jnp.dot(params, t_pows)


print(jnp.mean(jnp.abs(analytic_pred(plot_ts) - pred_soln)))


ax.plot(plot_ts, pred_soln,
        label="pred: z(t) = %.2ft + %.2ft^2 + %.2ft^3 + %.2ft^4" % (params[0], params[1], params[2], params[3]),
        c="orange")
ax.plot(plot_ts, true_soln, '--', label="true: z(t) = t^2",
        c="blue")
ax.scatter(ts, learned_soln, label="training points: MSE = %.2e" % loss_, c="blue")

# font = {'family':   'normal',
#         'weight':   'bold',
#         'size':     14}
# plt.rc('font', **font)
# plt.rc('text', usetex=True)

plt.legend()
plt.xlabel("t")
plt.ylabel("z(t)")

plt.title("Regularizing %.2f * d^3z(t)/dt^3, NFE: %d, init_step: 0.1" % (lam, nfe))
plt.savefig("dyn.png")
plt.clf()
plt.close(fig)
