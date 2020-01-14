"""
Neural ODEs on MNIST (only parameters are dynamics).

Example for profililng.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import pickle
import sys
import time

import numpy.random as npr
from jax.examples import datasets

import jax
import jax.numpy as np
from jax import random, grad, jet
from jax.experimental import optimizers
from jax.experimental.ode import build_odeint, odeint, vjp_odeint
from jax.flatten_util import ravel_pytree
from jax.nn import log_softmax
from jax.nn.initializers import glorot_normal, normal

REGS = ['none', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r45']

parser = argparse.ArgumentParser('ODE MNIST')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nepochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--mom', type=float, default=0.9)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=REGS, default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--dirname', type=str, default='tmp15')
parse_args = parser.parse_args()


assert os.path.exists(parse_args.dirname)

img_dim = 784
ode_dim = 20
num_reg = 1
n_classes = 10


reg = parse_args.reg
lam = parse_args.lam
rng = random.PRNGKey(0)
dirname = parse_args.dirname


# set up data
t = np.array([0., 1.])
REG_IND = REGS.index(reg) - 3

# set up dims to select
ode_in_dims = random.shuffle(rng, np.arange(img_dim))[:ode_dim]


# set up jet interface
def jet_wrap_predict(params):
    """
    Function which returns a closure that has the correct API for jet.
    """
    return lambda z, t: dynamics_predict(params, append_time(z, t))


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
  return append_aug(y, np.zeros((y.shape[0], num_reg)))


def unpack_aug(yr):
  """
  yr::(BS,D+R)
  y::(BS,D)
  r::(BS,R)
  """
  return yr[:, :ode_dim], yr[:, ode_dim:]


# set up model
def sigmoid(z):
    """
    Defined using only numpy primitives (but probably less numerically stable).
    """
    return 1./(1. + np.exp(-z))


def dynamics_predict(params, y_t):
    """
    MLP for dynamics of ODE.
    """
    w_dyn, b_dyn = params
    return sigmoid(np.dot(y_t, w_dyn) + np.expand_dims(b_dyn, axis=0))


def pre_ode(x):
    """
    Select subset of dims for ODE (since hard to integrate high-dims).
    """
    return x[:, ode_in_dims]


def post_ode(out_ode):
    """
    Select subset of output dims for classification prediction.
    """
    return log_softmax(out_ode[:, :10])


# set up ODE
def dynamics(flat_y, t, *flat_params):
    """
    Dynamics of the ODEBlock.
    """
    y = np.reshape(flat_y, (-1, ode_dim))
    dydt = dynamics_predict(ravel_ode_params(np.array(flat_params)), append_time(y, t))
    return np.ravel(dydt)


def reg_dynamics(y, t, params):
  """
  Computes regularization dynamics.
  """
  if reg == "none":
    regularization = np.zeros_like(y)
  elif reg == "r0":
    regularization = y
  elif reg == "r1":
    regularization = dynamics_predict(params, append_time(y, t))
  else:
    y0, y_n = sol_recursive(jet_wrap_predict(params), y, t)
    if reg == "r1":
      regularization = y0
    elif reg == 'r45':
        return np.sum(y_n[3] ** 2 - y_n[2] ** 2, axis=1, keepdims=True)
    else:
      regularization = y_n[REG_IND]
  return np.sum(regularization ** 2, axis=1, keepdims=True)


def aug_dynamics(flat_yr, t, *flat_params):
    """
    Dynamics augmented with regularization of the ODEBlock.
    """
    params = ravel_ode_params(np.array(flat_params))
    y, r = unpack_aug(np.reshape(flat_yr, (-1, ode_dim + num_reg)))
    dydt = dynamics_predict(params, append_time(y, t))
    drdt = reg_dynamics(y, t, params)
    return np.ravel(append_aug(dydt, drdt))


def initialize(rng):
    """
    Initialize parameters of the model.
    """
    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_1, b_1 = glorot_normal()(k1, (img_dim, ode_dim)), normal()(k2, (ode_dim,))

    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_dyn, b_dyn = glorot_normal()(k1, (ode_dim + 1, ode_dim)), normal()(k2, (ode_dim,))

    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_2 = glorot_normal()(k1, (ode_dim, n_classes))

    flat_ode_params, ravel_ode_params = ravel_pytree((w_dyn, b_dyn))
    init_params = [(w_1, b_1), flat_ode_params, (w_2,)]

    return rng, init_params, ravel_ode_params


# define ravel
_, _init_params, ravel_ode_params = initialize(rng)
_, ravel_params = ravel_pytree(_init_params)


def run(rng):
    """
    Run the Neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    train_images, train_labels, test_images, test_labels = datasets.mnist()

    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, parse_args.batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        """
        Stream of MNIST data.
        """
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * parse_args.batch_size:(i + 1) * parse_args.batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    nodeint = build_odeint(dynamics)
    nodeint_aug = build_odeint(aug_dynamics)

    def predict(args, x):
      """
      The prediction function for our model.
      """
      params = ravel_params(args)
      flat_ode_params = params[1]

      out_1 = pre_ode(x)

      flat_out_ode = nodeint_aug(np.ravel(aug_init(out_1)), t, *flat_ode_params)[-1]
      out_ode, out_ode_r = unpack_aug(np.reshape(flat_out_ode, (-1, ode_dim + num_reg)))

      out = post_ode(out_ode)

      return out, out_ode_r

    # set up loss
    @jax.jit
    def loss_fun(preds, targets):
        """
        Negative log-likelihood.
        """
        return -np.mean(np.sum(preds * targets, axis=1))

    @jax.jit
    def reg_loss_fun(reg):
        """
        Regularization loss function.
        """
        return np.mean(reg)

    @jax.jit
    def loss(params, batch):
        """
        Unaugmented loss to do GD over.
        """
        inputs, targets = batch
        preds, _ = predict(params, inputs)
        return loss_fun(preds, targets)

    @jax.jit
    def loss_aug(params, batch):
        """
        Augmented loss to do GD over.
        """
        inputs, targets = batch
        preds, reg = predict(params, inputs)
        return loss_fun(preds, targets) + lam * reg_loss_fun(reg)

    @jax.jit
    def sep_losses(opt_state, batch):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(opt_state)
        inputs, targets = batch
        pred, reg = predict(params, inputs)
        loss_ = loss_fun(pred, targets)
        reg_ = reg_loss_fun(reg)
        return loss_ + lam * reg_, loss_, reg_

    @jax.jit
    def partial_loss(out_ode, targets, args):
        """
        Evaluates loss wrt output of ODEBlock. (for b-NFE calculations).
        """
        preds = post_ode(out_ode)
        return loss_fun(preds, targets)

    @jax.jit
    def accuracy(params, batch):
        """
        Classification accuracy of the model.
        """
        inputs, targets = batch
        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(predict(params, inputs), axis=1)
        return np.mean(predicted_class == target_class)

    # optimizer
    rng, init_params, _ = initialize(rng)
    flat_params, ravel_params = ravel_pytree(init_params)
    opt_init, opt_update, get_params = optimizers.momentum(parse_args.lr, mass=parse_args.mom)
    opt_state = opt_init(flat_params)

    @jax.jit
    def update(i, opt_state, batch):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(i, grad(loss_aug)(get_params(opt_state), batch), opt_state)

    # unregularized system for counting NFE
    unreg_nodeint = jax.jit(lambda y0, t, args: odeint(dynamics, y0, t, *args))
    unreg_nodeint_vjp = jax.jit(lambda cotangent, y0, t, args:
                                vjp_odeint(dynamics, y0, t, *args, nfe=True)[1]
                                (np.reshape(cotangent,
                                            (parse_args.batch_time, cotangent.size // parse_args.batch_time)))[-1])

    @jax.jit
    def count_nfe(opt_state, batch):
        """
        Count NFE.
        """
        inputs, targets = batch
        params = ravel_params(get_params(opt_state))
        flat_ode_params = params[1]

        out_1 = pre_ode(inputs)

        in_ode = np.ravel(out_1)
        flat_out_ode, f_nfe = unreg_nodeint(in_ode, t, flat_ode_params)
        out_ode = np.reshape(flat_out_ode[-1], (-1, ode_dim))

        grad_partial_loss_ = grad(partial_loss)(out_ode, targets, params[2])
        # grad is 0 at t0 (since always equal)
        cotangent = np.stack((np.zeros_like(grad_partial_loss_), grad_partial_loss_), axis=0)
        b_nfe = unreg_nodeint_vjp(cotangent, np.reshape(in_ode, (-1, )), t, flat_ode_params)

        return f_nfe, b_nfe

    itercount = itertools.count()
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(batches)
            itr = next(itercount)

            nfe_start = time.time()
            f_nfe, b_nfe = count_nfe(opt_state, batch)
            nfe_time = time.time() - nfe_start

            print("forward NFE: %d" % f_nfe)
            print("backward NFE: %d" % b_nfe)

            update_start = time.time()
            opt_state = update(itr, opt_state, batch)
            update_time = time.time() - update_start

            print("nfe: %.5f, update: %.5f" % (nfe_time, update_time))

            if itr % parse_args.test_freq == 0:

                loss_aug_, loss_, loss_reg_ = sep_losses(opt_state, (train_images, train_labels))

                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f}'.format(itr, loss_aug_, loss_, loss_reg_)

                print(print_str)
                print(print_str, file=sys.stderr)

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()


if __name__ == "__main__":
    run(rng)
