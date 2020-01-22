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
parser.add_argument('--all_reg', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', False)


assert os.path.exists(parse_args.dirname)

reg = parse_args.reg
lam = parse_args.lam
rng = random.PRNGKey(0)
dirname = parse_args.dirname


# set up data
D = 1
NUM_REG = 1
REG_IND = REGS.index(reg) - 3
true_y0_range = 3
# only for evaluating on a fixed test set
true_y0 = np.repeat(
    np.expand_dims(np.linspace(-true_y0_range, true_y0_range, parse_args.data_size), axis=1), D, axis=1)  # (N, D)
ts = np.array([0., 1.])


# set up jet interface
def jet_wrap_predict(params):
  """
  Function which returns a closure that has the correct API for jet.
  """
  return lambda z, t: mlp_predict(params, append_time(z, t))


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


# set up utility functions
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


def append_aug(y, l, r):
  """
  y::(BS,D)
  r::(BS,R)
  yr::(BS,D+R)
  """
  ylr = np.concatenate((y, l, r), axis=1)
  return ylr


def append_all_aug(y, *r):
  """
  y::(BS,D)
  r::(BS,R)
  yr::(BS,D+R)
  """
  yr = np.concatenate((y, *r), axis=1)
  return yr


def aug_init(y):
  """
  Append 0s to get initial state of augmented system.
  """
  return append_aug(y, np.zeros((y.shape[0], 1)), np.zeros((y.shape[0], NUM_REG)))


def all_aug_init(y):
  """
  Append 0s to get initial state of augmented system.
  """
  return append_aug(y, np.zeros((y.shape[0], len(REGS) - 1)))


def unpack_aug(ylr):
  """
  yr::(BS,D+R)
  y::(BS,D)
  r::(BS,R)
  """
  return ylr[:, :D], ylr[:, D:D+1], ylr[:, D+1:]


# set up model
def sigmoid(z):
  """
  Defined using only numpy primitives (but probably less numerically stable).
  """
  return 1./(1. + np.exp(-z))


def mlp_predict(params, y_t):
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
def dynamics(y, t, *args):
  """
  y::unravel((BS,D))
  t::scalar
  args::flat_params
  """
  y = np.reshape(y, (-1, D))
  dydt = mlp_predict(ravel_params(np.array(args)), append_time(y, t))
  return np.ravel(dydt)


def true_dynamics(y, t):
  """
  The ground truth dynamics function.
  """
  return np.ones_like(y) * t ** 2


def loss_dynamics(y, t, params):
  """
  Dynamics for computing loss.
  """
  pred_fn = mlp_predict(params, append_time(y, t))
  return np.sum((pred_fn - true_dynamics(y, t)) ** 2, axis=1, keepdims=True)


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
        regularization = y_n[3] - y_n[2]
    else:
      regularization = y_n[REG_IND]
  return np.sum(regularization ** 2, axis=1, keepdims=True)


def all_reg_dynamics(y, t, params):
  """
  Computes all regularization dynamics.
  """
  y0, y_n = sol_recursive(jet_wrap_predict(params), y, t)
  regs = (y, y0, *(y_n[i - 3] for i in range(3, len(REGS) - 1)))
  ret = (np.sum(reg ** 2, axis=1, keepdims=True) for reg in regs)
  return (*ret, np.sum(y_n[3] ** 2 - y_n[2] ** 2, axis=1, keepdims=True))


def aug_dynamics(yr, t, *args):
  """
  yr::unravel((BS,D+R))
  t::scalar
  args::flat_params
  return::unravel((BS,D+R))
  """
  params = ravel_params(np.array(args))
  y, l, r = unpack_aug(np.reshape(yr, (-1, D + 1 + NUM_REG)))
  dydt = mlp_predict(params, append_time(y, t))
  dldt = loss_dynamics(y, t, params)
  drdt = reg_dynamics(y, t, params)
  return np.ravel(append_aug(dydt, dldt, drdt))


def all_aug_dynamics(yr, t, *args):
  """
  Augmented dynamics for computing all regularization.
  """
  params = ravel_params(np.array(args))
  y, r = unpack_aug(np.reshape(yr, (-1, D + len(REGS) - 1)))
  dydt = mlp_predict(params, append_time(y, t))
  drdt = all_reg_dynamics(y, t, params)
  return np.ravel(append_all_aug(dydt, *drdt))


def initialize(rng, hidden_dim):
  """
  Initialize the parameters of the MLP.
  """
  rng, layer_rng = random.split(rng)
  k1, k2 = random.split(layer_rng)
  w_1, b_1 = glorot_normal()(k1, (D + 1, hidden_dim)), normal()(k2, (hidden_dim,))

  rng, layer_rng = random.split(rng)
  k1, k2 = random.split(layer_rng)
  w_2, b_2 = glorot_normal()(k1, (hidden_dim, D)), normal()(k2, (D,))

  init_params = [(w_1, b_1), (w_2, b_2)]

  return rng, init_params


@jax.jit
def get_batch(key):
  """
  Get batch.
  """
  key, subkey = random.split(key)
  batch_y0 = random.uniform(subkey, (parse_args.batch_size, D),
                            minval=-true_y0_range, maxval=true_y0_range)              # (M, D)
  return key, batch_y0


# define ravel
_, ravel_params = ravel_pytree(initialize(rng, 50)[1])


def run(rng):
    """
    Run the Neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    nodeint = build_odeint(dynamics)
    nodeint_aug = build_odeint(aug_dynamics)
    nodeint_all_aug = build_odeint(all_aug_dynamics)

    # set up loss
    @jax.jit
    def loss_fun(l):
      """
      Mean squared error.
      """
      return np.mean(l)

    @jax.jit
    def reg_loss_fun(r_pred):
      """
      Mean.
      """
      return np.mean(r_pred)

    @jax.jit
    def all_reg_loss_fun(rs_pred):
      """
      Mean.
      """
      return np.mean(rs_pred, axis=0)

    @jax.jit
    def loss_reg(flat_params, y0, ts):
      """
      Regularization loss.
      """
      yrs_pred = nodeint_aug(np.ravel(aug_init(y0)), ts, *flat_params)
      y1_pred, l, r_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + 1 + NUM_REG)))
      return reg_loss_fun(r_pred)

    @jax.jit
    def loss_aug(flat_params, y0, ts):
      """
      Loss for augmented dynamics.
      """
      yrs_pred = nodeint_aug(np.ravel(aug_init(y0)), ts, *flat_params)
      y1_pred, l, r_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + 1 + NUM_REG)))
      return loss_fun(l) + lam * reg_loss_fun(r_pred)

    @jax.jit
    def sep_losses(flat_params, y0, ts):
      """
      Convenience function for calculating losses separately.
      """
      yrs_pred = nodeint_aug(np.ravel(aug_init(y0)), ts, *flat_params)
      y1_pred, l, r_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + 1 + NUM_REG)))
      reg_ = reg_loss_fun(r_pred)
      return loss_fun(l) + lam * reg_, loss_fun(l), reg_

    @jax.jit
    def sep_losses_all_reg(flat_params, ys, ts):
      """
      Convenience function for calculating losses separately (and all reg).
      """
      y0, y1_target = ys
      yrs_pred = nodeint_all_aug(np.ravel(all_aug_init(y0)), ts, *flat_params)
      y1_pred, rs_pred = unpack_aug(np.reshape(yrs_pred[-1], (-1, D + len(REGS) - 1)))
      loss_ = loss_fun(y1_pred, y1_target)
      regs_ = all_reg_loss_fun(rs_pred)
      return loss_, tuple(regs_)

    rng, init_params = initialize(rng, 50)
    flat_params, _ = ravel_pytree(init_params)
    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-2, gamma=0.99)
    opt_state = opt_init(flat_params)

    @jax.jit
    def update(itr, opt_state, batch_y0):
      """
      Update the parameters.
      """
      dldp = grad(loss_aug)(get_params(opt_state), batch_y0, ts)
      return opt_update(itr, dldp, opt_state)

    # counting NFE
    unreg_nodeint = jax.jit(lambda y0, t, args: odeint(dynamics, y0, t, *args))

    @jax.jit
    def count_nfe(opt_state, y0):
      """
      Count NFE and print.
      """
      flat_params = get_params(opt_state)
      flat_y0 = np.ravel(y0)

      ys_pred, f_nfe = unreg_nodeint(flat_y0, ts, flat_params)

      return f_nfe, 0

    itr = 0
    assert parse_args.data_size % parse_args.batch_size == 0
    batch_per_epoch = parse_args.data_size // parse_args.batch_size
    for epoch in range(parse_args.nepochs):
      for batch in range(batch_per_epoch):
        itr += 1

        rng, batch_y0 = get_batch(rng)

        f_nfe, b_nfe = count_nfe(opt_state, batch_y0)

        print("forward NFE: %d" % f_nfe)
        print("backward NFE: %d" % b_nfe)

        opt_state = update(itr, opt_state, batch_y0)

        if itr % parse_args.test_freq == 0:
            flat_params = get_params(opt_state)
            if parse_args.all_reg:
                loss_, regs_ = sep_losses_all_reg(flat_params, true_y0, ts)
                print_str = 'Iter {:04d} | Loss {:.6f} | %s' % \
                            " | ".join(("%s {:.6f}" % reg_name for reg_name in REGS[1:]))
                print_str = print_str.format(itr, loss_, *regs_)

                print(print_str)
                print(print_str, file=sys.stderr)
            else:
                loss_aug_, loss_, loss_reg_ = sep_losses(flat_params, true_y0, ts)

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
  run(rng)
