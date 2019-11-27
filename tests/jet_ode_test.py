# Copyright 2018 Google LLC
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

"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.

Integrate systems of ordinary differential equations (ODEs) using the JAX
autograd/diff library and the Dormand-Prince method for adaptive integration
stepsize calculation. Provides improved integration accuracy over fixed
stepsize integration methods.

Adjoint algorithm based on Appendix C of https://arxiv.org/pdf/1806.07366.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax.experimental.ode import build_odeint
from jax import random, grad, jet
from jax.nn.initializers import glorot_normal, normal
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import numpy as onp


def test_nodes_grad():
  """Compare numerical and exact differentiation of a Neural ODE."""
  data_size = 1000
  batch_size = 1
  batch_time = 2

  REGS = ['r2']
  NUM_REGS = len(REGS)

  reg = "r2"

  lam = 1

  D = 1

  rng = random.PRNGKey(0)

  true_y0_range = 3
  true_fn = lambda x: x ** 3
  # only for evaluating on a fixed test set
  true_y0 = np.repeat(np.expand_dims(np.linspace(-3, 3, data_size), axis=1), D, axis=1)  # (N, D)
  true_y1 = np.repeat(np.expand_dims(true_fn(true_y0[:, 0]), axis=1), D, axis=1)
  true_y = np.concatenate((np.expand_dims(true_y0, axis=0),
                          np.expand_dims(true_y1, axis=0)),
                          axis=0)  # (T, N, D)
  t = np.array([0., 1.])  # (T)

  def sigmoid(z):
    return 1./(1. + np.exp(-z))

  def predict(params, y_t):
    w_1, b_1 = params[0]
    z_1 = np.dot(y_t, w_1) + np.expand_dims(b_1, axis=0)
    out_1 = sigmoid(z_1)

    w_2, b_2 = params[1]
    z_1 = np.dot(out_1, w_2) + np.expand_dims(b_2, axis=0)

    return z_1

  def jet_wrap_predict(params):
    """
    Function which returns a closure that has the correct API for jet.
    """
    return lambda z, t: predict(params, np.concatenate((z, np.ones((z.shape[0], 1)) * t), axis=1))

  hidden_dim = 2

  # initialize the parameters
  rng, layer_rng = random.split(rng)
  k1, k2 = random.split(layer_rng)
  w_1, b_1 = glorot_normal()(k1, (D + 1, hidden_dim)), normal()(k2, (hidden_dim,))

  rng, layer_rng = random.split(rng)
  k1, k2 = random.split(layer_rng)
  w_2, b_2 = glorot_normal()(k1, (hidden_dim, D)), normal()(k2, (D,))

  init_params = [(w_1, b_1), (w_2, b_2)]

  # define ravel objects

  flat_params, ravel_params = ravel_pytree(init_params)

  @jax.jit
  def get_batch(key):
    """
    Get batch.
    """
    new_key, subkey = random.split(key)
    key = new_key
    batch_y0 = random.uniform(subkey, (batch_size, D),
                              minval=-true_y0_range, maxval=true_y0_range)              # (M, D)
    batch_y1 = np.repeat(np.expand_dims(true_fn(batch_y0[:, 0]), axis=1), D, axis=1)    # (M, D)
    batch_t = t                                                                         # (T)
    batch_y = np.concatenate((np.expand_dims(batch_y0, axis=0),
                              np.expand_dims(batch_y1, axis=0)),
                             axis=0)                                                    # (T, M, D)
    return key, batch_y0, batch_t, batch_y

  rng, batch_y0, batch_t, batch_y = get_batch(rng)

  r0 = np.zeros((batch_size, 1))
  allr0 = np.zeros((batch_size, NUM_REGS))
  r = np.zeros((batch_time, batch_size, 1))
  allr = np.zeros((batch_time, batch_size, NUM_REGS))
  test_r = np.zeros((batch_time, data_size, 1))
  test_allr = np.zeros((batch_time, data_size, NUM_REGS))
  test_r0 = np.zeros((data_size, 1))
  test_allr0 = np.zeros((data_size, NUM_REGS))

  batch_y0_t = np.concatenate((batch_y0,
                               np.expand_dims(
                                   np.repeat(batch_t[0], batch_size),
                                   axis=1)
                               ),
                              axis=1)
  # parse_args.batch_size * (D + 1) |-> (parse_args.batch_size, D + 1)
  _, ravel_batch_y0_t = ravel_pytree(batch_y0_t)

  batch_y0_t_r0_allr0 = np.concatenate((batch_y0_t, r0, allr0), axis=1)
  # parse_args.batch_size * (D + 2 + NUM_REGS) |-> (parse_args.batch_size, D + 2 + NUM_REGS)
  flat_batch_y0_t_r0_allr0, ravel_batch_y0_t_r0_allr0 = ravel_pytree(batch_y0_t_r0_allr0)

  batch_y_t_r_allr = np.concatenate((batch_y,
                                     np.expand_dims(
                                         np.tile(batch_t, (batch_size, 1)).T,
                                         axis=2),
                                     r,
                                     allr),
                                    axis=2)
  # parse_args.batch_time * parse_args.batch_size * (D + 2 + NUM_REGS) |->
  #                                                   (parse_args.batch_time, parse_args.batch_size, D + 2 + NUM_REGS)
  _, ravel_batch_y_t_r_allr = ravel_pytree(batch_y_t_r_allr)

  true_y_t_r_allr = np.concatenate((true_y,
                                    np.expand_dims(
                                        np.tile(batch_t, (data_size, 1)).T,
                                        axis=2),
                                    test_r,
                                    test_allr),
                                   axis=2)
  # parse_args.batch_time * parse_args.data_size * (D + 2 + NUM_REGS) |->
  #                                       (parse_args.batch_time, parse_args.data_size, D + 2 + NUM_REGS)
  _, ravel_true_y_t_r_allr = ravel_pytree(true_y_t_r_allr)

  true_y0_t_r0_allr = np.concatenate((true_y0,
                                      np.expand_dims(
                                          np.repeat(t[0], data_size), axis=1),
                                      test_r0,
                                      test_allr0), axis=1)
  # parse_args.data_size * (D + 2 + NUM_REGS) |-> (parse_args.data_size, D + 2 + NUM_REGS)
  flat_true_y0_t_r0_allr, ravel_true_y0_t_r0_allr = ravel_pytree(true_y0_t_r0_allr)

  fargs = flat_params

  @jax.jit
  def dynamics(y_t, t, *args):
      """
      Time-augmented dynamics.
      """

      flat_params = args
      params = ravel_params(np.array(flat_params))

      y_t = ravel_batch_y0_t(y_t)

      predictions_y = predict(params, y_t)
      predictions = np.concatenate((predictions_y,
                                    np.ones((batch_size, 1))),
                                   axis=1)

      flat_predictions = np.reshape(predictions, (-1,))
      return flat_predictions

  @jax.jit
  def reg_dynamics(y_t_r_allr, t, *args):
      """
      Augmented dynamics to implement regularization.
      """

      flat_params = args
      params = ravel_params(np.array(flat_params))

      # separate out state from augmented
      y_t_r_allr = ravel_batch_y0_t_r0_allr0(y_t_r_allr)
      y_t = y_t_r_allr[:, :D + 1]
      y = y_t[:, :-1]

      predictions_y = predict(params, y_t)
      predictions = np.concatenate((predictions_y,
                                    np.ones((batch_size, 1))),
                                   axis=1)

      y0, y_n = sol_recursive(jet_wrap_predict(params), y, y_t[:, -1][0])

      none = np.zeros(batch_size)
      r0 = np.sum(y ** 2, axis=1) ** 0.5
      r1 = np.sum(predictions_y ** 2, axis=1)
      r2 = np.sum(y_n[0] ** 2, axis=1)
      r3 = np.sum(y_n[1] ** 2, axis=1)
      r4 = np.sum(y_n[2] ** 2, axis=1)
      r5 = np.sum(y_n[3] ** 2, axis=1)
      r6 = np.sum(y_n[4] ** 2, axis=1)

      regs_map = {
          "none": none,
          # "r0": r0,
          # "r1": r1,
          "r2": r2,
          # "r3": r3,
          # "r4": r4,
          # "r5": r5,
          # "r6": r6
      }

      regularization = regs_map[reg]

      regs = tuple(np.expand_dims(regs_map[reg_name], axis=1) for reg_name in REGS)

      pred_reg = np.concatenate((predictions,
                                 np.expand_dims(regularization, axis=1),
                                 *regs),
                                axis=1)
      flat_pred_reg = np.reshape(pred_reg, (-1,))
      return flat_pred_reg

  @jax.jit
  def test_reg_dynamics(y_t_r_allr, t, *args):
      """
      Augmented dynamics to implement regularization. (on test)
      """

      flat_params = args
      params = ravel_params(np.array(flat_params))

      # separate out state from augmented
      # only difference between this and reg_dynamics is
      # ravelling over datasize instead of batch size
      y_t_r_allr = ravel_true_y0_t_r0_allr(y_t_r_allr)
      y_t = y_t_r_allr[:, :D + 1]
      y = y_t[:, :-1]

      predictions_y = predict(params, y_t)
      predictions = np.concatenate((predictions_y,
                                    np.ones((data_size, 1))),
                                   axis=1)

      y0, y_n = sol_recursive(jet_wrap_predict(params), y, y_t[:, -1][0])

      none = np.zeros(data_size)
      r0 = np.sum(y ** 2, axis=1) ** 0.5
      r1 = np.sum(predictions_y ** 2, axis=1)
      r2 = np.sum(y_n[0] ** 2, axis=1)
      r3 = np.sum(y_n[1] ** 2, axis=1)
      r4 = np.sum(y_n[2] ** 2, axis=1)
      r5 = np.sum(y_n[3] ** 2, axis=1)
      r6 = np.sum(y_n[4] ** 2, axis=1)

      regs_map = {
          "none": none,
          # "r0": r0,
          # "r1": r1,
          "r2": r2,
          # "r3": r3,
          # "r4": r4,
          # "r5": r5,
          # "r6": r6
      }

      regularization = regs_map[reg]

      regs = tuple(np.expand_dims(regs_map[reg_name], axis=1) for reg_name in REGS)

      pred_reg = np.concatenate((predictions,
                                 np.expand_dims(regularization, axis=1),
                                 *regs),
                                axis=1)
      flat_pred_reg = np.reshape(pred_reg, (-1,))
      return flat_pred_reg

  @jax.jit
  def total_loss_fun(pred_y_t_r, target):
      """
      Loss function.
      """
      pred, reg = pred_y_t_r[:, :, :D], pred_y_t_r[:, :, D + 1]
      return loss_fun(pred, target) + lam * reg_loss(reg)

  @jax.jit
  def reg_loss(reg):
      """
      Regularization loss function.
      """
      return np.mean(reg)

  @jax.jit
  def loss_fun(pred, target):
      """
      Mean squared error.
      """
      return np.mean((pred - target) ** 2)

  @jax.jit
  def nodes_predict(args):
      """
      Evaluate loss on model's predictions.
      """
      true_ys, odeint_args = args[0], args[1:]
      ys = ravel_batch_y_t_r_allr(nodes_odeint(*odeint_args))
      return total_loss_fun(ys, true_ys)

  nodes_odeint = build_odeint(reg_dynamics, atol=1e-12, rtol=1e-12)

  numerical_grad = nd(nodes_predict, (true_y, flat_batch_y0_t_r0_allr0, t, *fargs))
  exact_grad, ravel_grad = ravel_pytree(grad(nodes_predict)((true_y, flat_batch_y0_t_r0_allr0, t, *fargs))[1:])

  exact_grad = ravel_grad(exact_grad)
  numerical_grad = ravel_grad(numerical_grad)

  tmp1 = np.abs(exact_grad[0] - numerical_grad[0])
  tmp2 = np.abs(exact_grad[1] - numerical_grad[1])
  tmp3 = np.abs(np.array(exact_grad[2:]) - np.array(numerical_grad[2:]))

  print("y0", tmp1.shape)
  print(np.min(tmp1))
  print(np.max(tmp1))
  print(np.median(tmp1))
  print(np.mean(tmp1))
  print(tmp1)
  print("t0, t1", tmp2.shape)
  print(np.min(tmp2))
  print(np.max(tmp2))
  print(np.median(tmp2))
  print(np.mean(tmp2))
  print(tmp2)
  print("params", tmp3.shape)
  print(np.min(tmp3))
  print(np.max(tmp3))
  print(np.median(tmp3))
  print(np.mean(tmp3))
  print(tmp3)

  # wrt y0
  assert np.allclose(exact_grad[0], numerical_grad[0])

  # wrt [t0, t1]
  assert np.allclose(exact_grad[1], numerical_grad[1])

  # wrt params (currently fails, but atol is still pretty good)
  assert np.allclose(np.array(exact_grad[2:]), np.array(numerical_grad[2:]))


def nd(f, x, eps=1e-5):
  flat_x, unravel = ravel_pytree(x)
  dim = len(flat_x)
  g = onp.zeros_like(flat_x)
  for i in range(dim):
    d = onp.zeros_like(flat_x)
    d[i] = eps
    g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
  return g


def sol_recursive(f,z,t):
  # closure to expand z
  def g(z):
    return f(z,t)

  # First coeffs computed recursively
  (y0, [y1h]) = jet(g, (z, ), ((np.ones_like(z), ), ))
  (y0, [y1, y2h]) = jet(g, (z, ), (( y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jet(g, (z, ), ((y0, y1, y2h), ))
  (y0, [y1, y2, y3, y4h]) = jet(g, (z, ), ((y0, y1, y2, y3h), ))
  (y0, [y1, y2, y3, y4, y5h]) = jet(g, (z, ),
                                    ((y0, y1, y2, y3, y4h), ))
  (y0, [y1, y2, y3, y4, y5,y6h]) = jet(g, (z, ),
                                    ((y0, y1, y2, y3, y4, y5h), ))


  return (y0, [y1, y2, y3, y4, y5])


if __name__ == '__main__':
  from jax.config import config
  config.update("jax_enable_x64", True)

  test_nodes_grad()
