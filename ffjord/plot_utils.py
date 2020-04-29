import jax.numpy as np
from jax.api import jit, vmap
from functools import partial
import matplotlib.pyplot as plt

# ========= Helper function for plotting. =========

@partial(jit, static_argnums=(0, 1, 2, 4))
def _mesh_eval(func, x_limits, y_limits, params, num_ticks):
    # Evaluate func on a 2D grid defined by x_limits and y_limits.
    x = np.linspace(*x_limits, num=num_ticks)
    y = np.linspace(*y_limits, num=num_ticks)
    X, Y = np.meshgrid(x, y)
    xy_vec = np.stack([X.ravel(), Y.ravel()]).T
    zs = vmap(func, in_axes=(0, None))(xy_vec, params)
    #print(np.sum(zs) * (x_limits[1] - x_limits[0])
    # * (y_limits[1] - y_limits[0]) / (num_ticks**2))
    return X, Y, zs.reshape(X.shape)

def mesh_eval(func, x_limits, y_limits, params, num_ticks=51):
    return _mesh_eval(func, x_limits, y_limits, params, num_ticks)
