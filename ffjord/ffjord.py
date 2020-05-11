# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FFJORD: Free-form Continuous Dynamics
   for Scalable Reversible Generative Models

   https://arxiv.org/abs/1810.01367"""


from functools import partial
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)

from jax.api import jit, grad, vmap, jvp, jacfwd, vjp
from jax.experimental.jet import jet
from jax import random
from jax.experimental import optimizers
from jax.experimental.ode import odeint
import jax.numpy as np
import jax.scipy.stats.norm as norm

from data import gen_pinwheel
from plot_utils import mesh_eval

# ========= Functions to define a neural network. =========

def init_random_params(scale, layer_sizes, rng=random.PRNGKey(0)):
  return [(scale * random.normal(rng, (m, n)),
           scale * random.normal(rng, (n,)))  # Todo: use different rng at each layer
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def mlp(params, input):
    # Evaluate a multi-layer perceptron on a single input vector.
    for w, b in params:
        output = np.dot(input, w) + b
        input = np.tanh(output)
    return output

# ======== Define the model ==============

def standard_normal_logpdf(x):
    # Evaluate a single point on a diagonal multivariate Gaussian.
    return np.sum(vmap(norm.logpdf, in_axes=(0, None, None))(x, 0., 1.))

def nn_dynamics(z, t, args):
    return mlp(args, np.hstack([z, t]))

def ffjord_sample(params, D, rng):
    z = random.normal(rng, D)
    return odeint(nn_dynamics, z, np.array([1., 0.]), params)  # Todo: reverse time

def ffjord_dynamics(ffjord_state, t, eps, args):
    z = ffjord_state[:-1]
    # Hutchinson's estimator of the trace of the Jacobian of f.
    # eps must be drawn from a distribution with zero mean and
    # identity covariance.
    f = lambda z: nn_dynamics(z, t, args)
    # dz_dt, jac_times_eps = jvp(f, (z,), (eps,))
    dz_dt, pullback = vjp(f,z)
    eps_times_jac = pullback(eps)
    # dz_dt = f(z)
    # dlogp_dt = np.dot(eps, jac_times_eps)
    dlogp_dt = np.dot(eps_times_jac[0], eps)
    # dz_dt = f(z)
    # J = jacfwd(f)(z)
    # dlogp_dt = np.trace(J)
    return np.hstack([dz_dt, -dlogp_dt])


def ffjord_log_density(params, x, D, rng):
    init_state = np.hstack([x, 0.])
    eps = random.normal(rng, (D,))
    ffjord_out = odeint(ffjord_dynamics, init_state, np.array([0., 1.]), eps, params)[1]
    z, dlogp = ffjord_out[:-1], ffjord_out[-1]
    logpz = standard_normal_logpdf(z)
    logpx = logpz - dlogp
    return logpx


def batch_likelihood(params, data, rng):
    N, D = np.shape(data)
    rngs = random.split(rng, N)
    batch_density = vmap(ffjord_log_density, in_axes=(None, 0, None, 0))
    return np.mean(batch_density(params, data, D, rngs))


# ========= Regularization Dynamics =========

def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  (y0, [y1h]) = jet(f, (z, t ), ((np.ones_like(z), ), (np.ones_like(t), )))
  (y0, [y1, y2h]) = jet(f, (z, t ), ((y0, y1h,), (np.ones_like(t), np.zeros_like(t),) ))
  (y0, [y1, y2, y3h]) = jet(f, (z, t ), ((y0, y1, y2h), (np.ones_like(t), np.zeros_like(t), np.zeros_like(t)) ))

  return (y0, [y1, y2])

def reg_dynamics(reg_aug_state, t, eps, params):
    ffjord_state, reg_state = reg_aug_state[:-1], reg_aug_state[-1]
    def _ffjord_dyn(z,t):
        return ffjord_dynamics(z, t, eps, params)
    def _reg_dyn(z,t):
        return sol_recursive(_ffjord_dyn, z,t)
    dz_dt, dnz_dtn = sol_recursive(_ffjord_dyn, ffjord_state, t)
    y1, y2 = dnz_dtn
    r3 = np.linalg.norm(y2)
    return np.hstack([dz_dt, r3])


def reg_log_density(params,x,D,rng, lam = 1):
    init_ffjord_state = np.hstack([x, 0.])
    init_reg_state = 0.
    init_aug_reg_state = np.hstack([init_ffjord_state,init_reg_state])

    eps = random.normal(rng, (D,))

    aug_reg_out = odeint(reg_dynamics, init_aug_reg_state, np.array([0.,1.]),eps,params)[1]

    z = aug_reg_out[:-2]
    dlogp = aug_reg_out[-2:-1]
    logpz = standard_normal_logpdf(z)
    logpx = logpz - dlogp

    reg_out = aug_reg_out[-1]

    return logpx - lam * reg_out

def batch_reg_likelihood(params, data, rng):
    N, D = np.shape(data)
    rngs = random.split(rng, N)
    batch_density = vmap(reg_log_density, in_axes=(None, 0, None, 0))
    return np.mean(batch_density(params, data, D, rngs))



# ========= Define an intractable unnormalized density =========

if __name__ == "__main__":

    data, _ = gen_pinwheel(num_classes=3)
    D = data.shape[1]

    @jit
    def objective(params, t):
        rng = random.PRNGKey(t)
        return -batch_likelihood(params, data, rng)

    @jit
    def reg_objective(params, t):
        rng = random.PRNGKey(t)
        return -batch_reg_likelihood(params, data, rng)

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    x_limits = [-2, 2]
    y_limits = [-2, 2]

    rng = random.PRNGKey(0)

    def test_grads():
        from jax.test_util import check_grads
        N, D = np.shape(data)
        rngs = random.split(rng, N)
        batch_density = vmap(ffjord_log_density, in_axes=(None, 0, None, 0))
        _vffjord = lambda p : batch_density(p,data,D,rngs)
        check_grads(_vffjord, (params,), 1, modes=["rev"])

    def callback(params, t):
        print("Iteration {} log-likelihood: {}".format(t, -objective(params, t)))
        print("Iteration {} reg-objective: {}".format(t, -reg_objective(params, t)))
        # print("testing grads")
        # test_grads()

        ffjord_dist = lambda x, params: np.exp(ffjord_log_density(params, x, D, rng))

        plt.cla()
        X, Y, Z = mesh_eval(ffjord_dist, x_limits, y_limits, params)
        ax.contour(X, Y, Z, cmap='summer')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_yticks([])
        ax.set_xticks([])

        ax.plot(data[:, 0], data[:, 1], 'b.')

        plt.draw()
        plt.pause(1.0/60.0)


    # Set up optimizer.
    init_params = init_random_params(0.1, [D + 1, 50, 100, D], rng)
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = grad(reg_objective)(params, i)
        return opt_update(i, gradient, opt_state)

    # Main loop.
    print("Optimizing...")
    for t in range(10000):
        opt_state = update(t, opt_state)
        params = get_params(opt_state)
        if t % 2 == 1: callback(params, t)
    plt.show(block=True)
