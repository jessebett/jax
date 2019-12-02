"""
Neural ODEs in Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as np
from jax import random, grad, jet
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.ode import build_odeint, odeint, vjp_odeint
from jax.flatten_util import ravel_pytree
from jax.nn.initializers import glorot_normal, normal

REGS = ['none', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r45']

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--nepochs', type=int, default=500)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=REGS, default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--dirname', type=str, default='tmp15')
parser.add_argument('--test', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', True)


def run(reg, lam, rng, dirname):
    """
    Run the Neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    # set up data
    D = 1
    NUM_REG = 1
    REG_IND = REGS.index(reg) - 3
    true_y0_range = 3
    true_fn = lambda x: x ** 3
    # only for evaluating on a fixed test set
    true_y0 = np.repeat(
        np.expand_dims(np.linspace(-true_y0_range, true_y0_range, parse_args.data_size), axis=1), D, axis=1)  # (N, D)
    true_y1 = np.repeat(np.expand_dims(true_fn(true_y0[:, 0]), axis=1), D, axis=1)
    true_y = (true_y0, true_y1)
    ts = np.array([0., 1.])

    @jax.jit
    def get_batch(key):
      """
      Get batch.
      """
      key, subkey = random.split(key)
      batch_y0 = random.uniform(subkey, (parse_args.batch_size, D),
                                minval=-true_y0_range, maxval=true_y0_range)              # (M, D)
      batch_y1 = np.repeat(np.expand_dims(true_fn(batch_y0[:, 0]), axis=1), D, axis=1)    # (M, D)
      return key, (batch_y0, batch_y1)

    def append_time(y, t):
      """
      y::(BS,D)
      t::scalar
      yt::(BS,D+1)
      """
      yt = np.concatenate((y,
                           np.ones((y.shape[0], 1)) * t),
                          axis=1)
      return yt

    # set up jet interface
    def jet_wrap_predict(params):
      """
      Function which returns a closure that has the correct API for jet.
      """
      return lambda z, t: predict(params, append_time(z, t))

    def sol_recursive(f, z, t):
      """
      Recursively compute higher order derivatives of dynamics of ODE.
      """
      def g(z):
        """
        Closure to expand z.
        """
        return f(z, t)

      # First coeffs computed recursively
      (y0, [y1h]) = jet(g, (z, ), ((np.ones_like(z), ), ))
      (y0, [y1, y2h]) = jet(g, (z, ), ((y0, y1h,), ))
      (y0, [y1, y2, y3h]) = jet(g, (z, ), ((y0, y1, y2h), ))
      (y0, [y1, y2, y3, y4h]) = jet(g, (z, ), ((y0, y1, y2, y3h), ))
      (y0, [y1, y2, y3, y4, y5h]) = jet(g, (z, ),
                                        ((y0, y1, y2, y3, y4h), ))
      (y0, [y1, y2, y3, y4, y5, y6h]) = jet(g, (z, ),
                                            ((y0, y1, y2, y3, y4, y5h), ))

      return (y0, [y1, y2, y3, y4, y5])

    # set up model
    def sigmoid(z):
      """
      Defined using only numpy primitives (but probably less numerically stable).
      """
      return 1./(1. + np.exp(-z))

    def predict(params, y_t):
      """
      MLP for dynamics of ODE.
      """
      w_1, b_1 = params[0]
      z_1 = np.dot(y_t, w_1) + np.expand_dims(b_1, axis=0)
      out_1 = sigmoid(z_1)

      w_2, b_2 = params[1]
      z_1 = np.dot(out_1, w_2) + np.expand_dims(b_2, axis=0)

      return z_1

    # set up ODE
    @jax.jit
    def dynamics(y, t, *args):
      """
      y::unravel((BS,D))
      t::scalar
      args::flat_params
      """
      y = np.reshape(y, (-1, D))
      dydt = predict(ravel_params(np.array(args)), append_time(y, t))
      return np.ravel(dydt)
    nodeint = build_odeint(dynamics)

    def append_aug(y, r):
      """
      y::(BS,D)
      r::(BS,R)
      yr::(BS,D+R)
      """
      yr = np.concatenate((y, r), axis=1)
      return yr

    def aug_init(y):
      """
      Append 0s to get initial state of augmented system.
      """
      return append_aug(y, np.zeros((y.shape[0], NUM_REG)))

    def unpack_aug(yr):
      """
      yr::(BS,D+R)
      y::(BS,D)
      r::(BS,R)
      """
      return yr[:, :D], yr[:, D:]

    @jax.jit
    def reg_dynamics(y, t, params):
      """
      Computes regularization dynamics.
      """
      if reg == "none":
        regularization = np.zeros_like(y)
      elif reg == "r0":
        regularization = y
      else:
        y0, y_n = sol_recursive(jet_wrap_predict(params), y, t)
        if reg == "r1":
          regularization = y0
        elif reg == 'r45':
            return np.sum(y_n[3] ** 2 - y_n[2] ** 2, axis=1, keepdims=True)
        else:
          regularization = y_n[REG_IND]
      return np.sum(regularization ** 2, axis=1, keepdims=True)

    @jax.jit
    def aug_dynamics(yr, t, *args):
      """
      yr::unravel((BS,D+R))
      t::scalar
      args::flat_params
      return::unravel((BS,D+R))
      """
      params = ravel_params(np.array(args))
      y, r = unpack_aug(np.reshape(yr, (-1, D + NUM_REG)))
      dydt = predict(params, append_time(y, t))
      drdt = reg_dynamics(y, t, params)
      return np.ravel(append_aug(dydt, drdt))
    nodeint_aug = build_odeint(aug_dynamics)

    # set up loss
    @jax.jit
    def loss_fun(pred, target):
      """
      Mean squared error.
      """
      return np.mean((pred - target) ** 2)

    @jax.jit
    def reg_loss_fun(r_pred):
      """
      Mean.
      """
      return np.mean(r_pred)

    @jax.jit
    def loss(flat_params, ys, ts):
      """
      Return function loss(params) for gradients
      """
      y0, y1_target = ys
      ys_pred = nodeint(np.ravel(y0), ts, *flat_params)
      y1_pred = np.reshape(ys_pred[-1], (-1, D))
      return loss_fun(y1_pred, y1_target)
    grad_loss_fun = grad(loss_fun)

    @jax.jit
    def loss_reg(flat_params, ys, ts):
      """
      Regularization loss.
      """
      y0, y1_target = ys
      yrs_pred = nodeint_aug(np.ravel(aug_init(y0)), ts, *flat_params)
      y1_pred, r_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + NUM_REG)))
      return reg_loss_fun(r_pred)

    @jax.jit
    def loss_aug(flat_params, ys, ts):
      """
      Loss for augmented dynamics.
      """
      y0, y1_target = ys
      yrs_pred = nodeint_aug(np.ravel(aug_init(y0)), ts, *flat_params)
      y1_pred, r_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + NUM_REG)))
      return loss_fun(y1_pred, y1_target) + lam * reg_loss_fun(r_pred)

    @jax.jit
    def sep_losses(flat_params, ys, ts):
      """
      Convenience function for calculating losses separately.
      """
      y0, y1_target = ys
      yrs_pred = nodeint_aug(np.ravel(aug_init(y0)), ts, *flat_params)
      y1_pred, r_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + NUM_REG)))
      loss_ = loss_fun(y1_pred, y1_target)
      reg_ = reg_loss_fun(r_pred)
      return loss_ + lam * reg_, loss_, reg_

    # initialize the parameters
    hidden_dim = 50
    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_1, b_1 = glorot_normal()(k1, (D + 1, hidden_dim)), normal()(k2, (hidden_dim,))

    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_2, b_2 = glorot_normal()(k1, (hidden_dim, D)), normal()(k2, (D,))

    init_params = [(w_1, b_1), (w_2, b_2)]

    # optimizer
    flat_params, ravel_params = ravel_pytree(init_params)
    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-2, gamma=0.99)
    opt_state = opt_init(flat_params)

    @jax.jit
    def update(itr, opt_state, batch_y):
      """
      Update the parameters.
      """
      dldp = grad(loss_aug)(get_params(opt_state), batch_y, ts)
      return opt_update(itr, dldp, opt_state)

    if parse_args.test:
      import numpy as onp

      def nd(f, x, eps=1e-5):
        """
        Numerical differentiation
        """
        flat_x, unravel = ravel_pytree(x)
        dim = len(flat_x)
        g = onp.zeros_like(flat_x)
        for i in range(dim):
          d = onp.zeros_like(flat_x)
          d[i] = eps
          g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
        return g

      _, batch_y = get_batch(rng)  # don't update rng, don't let test interfere w/ seed of training
      dr2dp = grad(loss_aug)(flat_params, batch_y, ts)
      dr2dp_numerical = nd(lambda flat_params: loss_aug(flat_params, batch_y, ts), flat_params)

      abs_diff = onp.abs(dr2dp - dr2dp_numerical)
      print("max abs. diff", onp.max(abs_diff))
      print("mean abs. diff", onp.mean(abs_diff))
      print("median abs. diff", onp.median(abs_diff))
      print(abs_diff)
      assert onp.allclose(dr2dp, dr2dp_numerical, atol=1e-5)

    # counting NFE
    unreg_nodeint = jax.jit(lambda y0, t, args: odeint(dynamics, y0, t, *args))
    unreg_nodeint_vjp = jax.jit(lambda cotangent, y0, t, args:
                                vjp_odeint(dynamics, y0, t, *args, nfe=True)[1](np.reshape(cotangent,
                                                                                           (parse_args.batch_time,
                                                                                            parse_args.batch_size *
                                                                                            D)))[-1])

    @jax.jit
    def count_nfe(opt_state, ys):
      """
      Count NFE and print.
      """
      y0, y1_target = ys
      flat_params = get_params(opt_state)
      flat_y0 = np.ravel(y0)

      ys_pred, f_nfe = unreg_nodeint(flat_y0, ts, flat_params)
      y1_pred = np.reshape(ys_pred[-1], (-1, D))

      grad_loss_fun_ = grad_loss_fun(y1_pred, y1_target)
      # grad is 0 at t0 (since always equal)
      cotangent = np.stack((np.zeros_like(grad_loss_fun_), grad_loss_fun_), axis=0)
      b_nfe = unreg_nodeint_vjp(cotangent, flat_y0, ts, flat_params)

      return f_nfe, b_nfe

    itr = 0
    assert parse_args.data_size % parse_args.batch_size == 0
    batch_per_epoch = parse_args.data_size // parse_args.batch_size
    for epoch in range(parse_args.nepochs):
      for batch in range(batch_per_epoch):
        itr += 1

        rng, batch_y = get_batch(rng)

        f_nfe, b_nfe = count_nfe(opt_state, batch_y)

        print("forward NFE: %d" % f_nfe)
        print("backward NFE: %d" % b_nfe)

        opt_state = update(itr, opt_state, batch_y)

        if itr % parse_args.test_freq == 0:
            flat_params = get_params(opt_state)
            loss_aug_, loss_, loss_reg_ = sep_losses(flat_params, true_y, ts)

            print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | ' \
                        'Loss {:.6f} | r {:.6f}'.format(itr, loss_aug_, loss_, loss_reg_)

            print(print_str)
            print(print_str, file=sys.stderr)

        if itr % parse_args.save_freq == 0:
            flat_params = get_params(opt_state)
            param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
            outfile = open(param_filename, "wb")
            pickle.dump(flat_params, outfile)
            outfile.close()


if __name__ == "__main__":
  assert os.path.exists(parse_args.dirname)
  run(parse_args.reg, parse_args.lam, random.PRNGKey(0), parse_args.dirname)
