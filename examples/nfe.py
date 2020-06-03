"""
Exploration of NFE for different solvers on certain orders of polynomials.
"""

import collections
import pickle
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.experimental.ode import ravel_first_arg, \
    _heun_odeint, _fehlberg_odeint,\
    _bosh_odeint, \
    _owrenzen_odeint,\
    _dopri5_odeint, _owrenzen5_odeint
from jax.flatten_util import ravel_pytree


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


def dyn(z, t, params):
    """
    Learning polynomial dynamics.
    """
    t_pows = jnp.array([(i + 1) * _int_pow(t, i) for i in range(dyn_degree)])
    return jnp.dot(t_pows, params)


dyn_degree = 6  # degree of predicted soln, x-1 degree of dynamics
y0 = 1.
ts = jnp.array([0., 100.])
y0, unravel = ravel_pytree(y0)
dyn = ravel_first_arg(dyn, unravel)

# name, method, order, stages
methods = [("heun", _heun_odeint, 2, 1),
           ("fehlberg", _fehlberg_odeint, 2, 2),
           ("bosh", _bosh_odeint, 3, 3),
           ("owrenzen", _owrenzen_odeint, 4, 5),
           # ("owrenzen5", _owrenzen5_odeint, 5, 7),
           ("dopri", _dopri5_odeint, 5, 6)]


init_nfe = 2    # 1 if step size given and not automatic

steps_filename = "steps.pickle"

try:
    steps_file = open(steps_filename, "rb")
    all_nsteps = pickle.load(steps_file)
    steps_file.close()
except IOError:
    all_nsteps = collections.defaultdict(dict)

    for degree in range(dyn_degree):
        # from t^0 to t^k, len() = dyn_degree
        params = jax.ops.index_update(jnp.zeros(dyn_degree), jax.ops.index[degree], 1)
        print(params)
        for name, _method, order, stage in methods:
            _, nfe = _method(dyn, 1.4e-8, 1.4e-8, jnp.inf, y0, ts, params)
            assert (nfe - init_nfe) % stage == 0
            n_steps = (nfe - init_nfe) // stage
            all_nsteps[name][degree] = n_steps
            print(name, order, n_steps)

    steps_file = open(steps_filename, "wb")
    pickle.dump(all_nsteps, steps_file)
    steps_file.close()

cm = plt.get_cmap('copper')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
plt.rc('text', usetex=True)
fig, ax = plt.subplots()

regs = ["r2", "r3", "r4", "r5"]
num_points = len(regs)
c_spacing = jnp.linspace(0, 1, num=num_points)
cmap = lambda ind: cm(c_spacing[ind])

print(all_nsteps)

for name, _, order, _ in methods:
    x, y = zip(*sorted(all_nsteps[name].items()))
    x = jnp.array(x)
    x += 1
    y = jnp.array(y)
    y = jnp.concatenate((jnp.array([0]), y[1:] - y[:-1]))
    y /= sum(y)
    ax.plot(x, y, label="%s%d" % (name, order), c=cmap(order))

ax.set_xlabel(r'$k : z(t) = \displaystyle\frac{t^k}{k}$')
ax.set_ylabel(r"$\Delta$ number of steps between $k$ and $k-1$ (normalized)")
plt.gcf().subplots_adjust(bottom=0.18)
plt.legend()
plt.savefig("solver_steps.png")
plt.savefig("solver_steps.pdf")
plt.clf()
plt.close(fig)
