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

from jax.api import jit, grad, vmap, jvp, jacfwd
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

def ffjord_log_density(params, x, D, rng):

    eps = random.normal(rng, (D,))
    def aug_dynamics(aug_state, t, args):
        z = aug_state[:-1]
        # Hutchinson's estimator of the trace of the Jacobian of f.
        # eps must be drawn from a distribution with zero mean and
        # identity covariance.
        f = lambda z: nn_dynamics(z, t, args)
        dz_dt, jac_times_eps = jvp(f, (z,), (eps,))
        # dz_dt = f(z)
        dlogp_dt = np.dot(eps, jac_times_eps)
        # dz_dt = f(z)
        # J = jacfwd(f)(z)
        # dlogp_dt = np.trace(J)
        return np.hstack([dz_dt, dlogp_dt])

    init_state = np.hstack([x, 0.])
    aug_out = odeint(aug_dynamics, init_state, np.array([0., 1.]), params)[1]
    z, dlogp = aug_out[:-1], aug_out[-1]
    logpz = standard_normal_logpdf(z)
    return logpz + dlogp


def batch_likelihood(params, data, rng):
    N, D = np.shape(data)
    rngs = random.split(rng, N)
    batch_density = vmap(ffjord_log_density, in_axes=(None, 0, None, 0))
    return np.mean(batch_density(params, data, D, rngs))



# ========= Define an intractable unnormalized density =========

if __name__ == "__main__":

    data, _ = gen_pinwheel()
    D = data.shape[1]

    @jit
    def objective(params, t):
        rng = random.PRNGKey(t)
        return -batch_likelihood(params, data, rng)

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    x_limits = [-2, 2]
    y_limits = [-2, 2]

    rng = random.PRNGKey(0)
    def callback(params, t):
        print("Iteration {} log-likelihood: {}".format(t, -objective(params, t)))

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
    opt_init, opt_update, get_params = optimizers.momentum(step_size=0.01, mass=0.9)
    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = grad(objective)(params, i)
        return opt_update(i, gradient, opt_state)

    # Main loop.
    print("Optimizing...")
    for t in range(10000):
        opt_state = update(t, opt_state)
        params = get_params(opt_state)
        if t % 50 == 1: callback(params, t)
    plt.show(block=True)
