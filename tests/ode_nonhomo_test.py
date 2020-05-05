from functools import reduce

from scipy.special import factorial as fact

import jax
import jax.numpy as np
from jax import lax
from jax.experimental.ode import odeint
from jax.experimental.jet import jet
from jax.config import config
config.update("jax_enable_x64", True)

def jvp_taylor(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  n_derivs = []
  N = len(series[0]) + 1
  for i in range(1, N):
    d = repeated(jax.jacobian, i)(expansion)(0.)
    n_derivs.append(d)
  return f(*primals), n_derivs

def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)

  return rfun

def sol_recursive(f, z_t, _t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  (y0, [y1h]) = jet(f, (z_t, _t ), ((np.ones_like(z_t), ), (np.ones_like(_t), )))
  (y0, [y1, y2h]) = jet(f, (z_t, _t ), ((y0, y1h,), (np.ones_like(_t), np.zeros_like(_t),) ))
  (y0, [y1, y2, y3h]) = jet(f, (z_t, _t ), ((y0, y1, y2h), (np.ones_like(_t), np.zeros_like(_t), np.zeros_like(_t)) ))
  (y0, [y1, y2, y3, y4h]) = jet(f, (z_t, _t ), ((y0, y1, y2, y3h), (np.ones_like(_t), np.zeros_like(_t), np.zeros_like(_t), np.zeros_like(_t))))
  (y0, [y1, y2, y3, y4, y5h]) = jet(f, (z_t, _t), ((y0, y1, y2, y3, y4h), (np.ones_like(_t), np.zeros_like(_t), np.zeros_like(_t), np.zeros_like(_t), np.zeros_like(_t))))

  return (y0, [y1, y2, y3, y4, y5h])

def test_sol_recursive():
  from scipy.integrate import odeint

  def f(z_t, t):
    z = z_t[0]
    v = z_t[1]
    dz = v
    dv = (7*t*v - 16 * z) / (t**2)
    return np.array([dz,dv])

  # Initial  Conditions
  t0 = np.array([1.])
  z0 = np.array([1., 4.])

  # Closed-form solution
  def true_sol(t):
    return np.array([t**4, 4*t**3])

  # Evaluate at t_eval
  t_eval = np.array([2.])
  z_eval_true = true_sol(t_eval)
  z_t_eval_true = np.concatenate((z_eval_true, t_eval))

  (y0, [y1, y2, y3, y4, y5h]) = sol_recursive(f, z_eval_true, t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, z0, [t0[0], t_eval[0]])[1]
  print(z_eval_true)
  print(z_eval_odeint)
  assert np.isclose(z_eval_true, z_eval_odeint)

  # True derivatives of solution
  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0.,0.,0.), ))

  return np.array([y0, y1, y2, y3, y4]), np.array(true_ds[1][:-2])



