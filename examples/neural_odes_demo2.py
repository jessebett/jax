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
from jax.experimental.ode import odeint, build_odeint, vjp_odeint
from jax.flatten_util import ravel_pytree
from jax.nn.initializers import glorot_normal, normal

REGS = ['r0', 'r1', 'r2']
NUM_REGS = len(REGS)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--nepochs', type=int, default=1000)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--dirname', type=str, default='tmp15')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', True)

D = 1
true_y0_range = 3
true_fn = lambda x: x ** 3
# only for evaluating on a fixed test set
true_y0 = np.repeat(np.expand_dims(np.linspace(-3, 3, parse_args.data_size), axis=1), D, axis=1)  # (N, D)
true_y1 = np.repeat(np.expand_dims(true_fn(true_y0[:, 0]), axis=1), D, axis=1)
true_y = np.concatenate((np.expand_dims(true_y0, axis=0),
                        np.expand_dims(true_y1, axis=0)),
                        axis=0)  # (T, N, D)
t = np.array([0., 1.])  # (T)


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

@jax.jit
def get_batch(key):
    """
    Get batch.
    """
    new_key, subkey = random.split(key)
    key = new_key
    batch_y0 = random.uniform(subkey, (parse_args.batch_size, D),
                              minval=-true_y0_range, maxval=true_y0_range)              # (M, D)
    batch_y1 = np.repeat(np.expand_dims(true_fn(batch_y0[:, 0]), axis=1), D, axis=1)    # (M, D)
    batch_t = t                                                                         # (T)
    batch_y = np.concatenate((np.expand_dims(batch_y0, axis=0),
                              np.expand_dims(batch_y1, axis=0)),
                             axis=0)                                                    # (T, M, D)
    return key, batch_y0, batch_t, batch_y


@jax.jit
def pack_batch(key):
    """
    Get batch and package it for augmented system integration.
    """
    key, batch_y0, batch_t, batch_y = get_batch(key)
    batch_y0_t = np.concatenate((batch_y0,
                                 np.expand_dims(
                                     np.repeat(batch_t[0], parse_args.batch_size),
                                     axis=1)
                                 ),
                                axis=1)
    flat_batch_y0_t = np.reshape(batch_y0_t, (-1,))
    r0 = np.zeros((parse_args.batch_size, 1))
    allr0 = np.zeros((parse_args.batch_size, NUM_REGS))
    batch_y0_t_r0_allr0 = np.concatenate((batch_y0_t, r0, allr0), axis=1)
    flat_batch_y0_t_r0_allr0 = np.reshape(batch_y0_t_r0_allr0, (-1,))
    return key, flat_batch_y0_t, flat_batch_y0_t_r0_allr0, batch_t, batch_y


def run(reg, lam, rng, dirname):
    """
    Run the neural ODEs method.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

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
        return lambda z, t: predict(params, np.concatenate((z,
                                                            np.ones((z.shape[0], 1)) * t),
                                                           axis=1))

    hidden_dim = 50

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

    rng, batch_y0, batch_t, batch_y = get_batch(rng)

    r0 = np.zeros((parse_args.batch_size, 1))
    allr0 = np.zeros((parse_args.batch_size, NUM_REGS))
    r = np.zeros((parse_args.batch_time, parse_args.batch_size, 1))
    allr = np.zeros((parse_args.batch_time, parse_args.batch_size, NUM_REGS))
    test_r = np.zeros((parse_args.batch_time, parse_args.data_size, 1))
    test_allr = np.zeros((parse_args.batch_time, parse_args.data_size, NUM_REGS))
    test_r0 = np.zeros((parse_args.data_size, 1))
    test_allr0 = np.zeros((parse_args.data_size, NUM_REGS))

    batch_y0_t = np.concatenate((batch_y0,
                                 np.expand_dims(
                                     np.repeat(batch_t[0], parse_args.batch_size),
                                     axis=1)
                                 ),
                                axis=1)
    # parse_args.batch_size * (D + 1) |-> (parse_args.batch_size, D + 1)
    _, ravel_batch_y0_t = ravel_pytree(batch_y0_t)

    batch_y_t = np.concatenate((batch_y,
                                np.expand_dims(
                                    np.tile(batch_t, (parse_args.batch_size, 1)).T,
                                    axis=2)
                                ),
                               axis=2)
    # parse_args.batch_time * parse_args.batch_size * (D + 1) |-> (parse_args.batch_time, parse_args.batch_size, D + 1)
    _, ravel_batch_y_t = ravel_pytree(batch_y_t)

    batch_y0_t_r0_allr0 = np.concatenate((batch_y0_t, r0, allr0), axis=1)
    # parse_args.batch_size * (D + 2 + NUM_REGS) |-> (parse_args.batch_size, D + 2 + NUM_REGS)
    _, ravel_batch_y0_t_r0_allr0 = ravel_pytree(batch_y0_t_r0_allr0)

    batch_y_t_r_allr = np.concatenate((batch_y,
                                       np.expand_dims(
                                           np.tile(batch_t, (parse_args.batch_size, 1)).T,
                                           axis=2),
                                       r,
                                       allr),
                                      axis=2)
    # parse_args.batch_time * parse_args.batch_size * (D + 2 + NUM_REGS) |->
    #                                                   (parse_args.batch_time, parse_args.batch_size, D + 2 + NUM_REGS)
    _, ravel_batch_y_t_r_allr = ravel_pytree(batch_y_t_r_allr)

    true_y_t_r_allr = np.concatenate((true_y,
                                      np.expand_dims(
                                          np.tile(batch_t, (parse_args.data_size, 1)).T,
                                          axis=2),
                                      test_r,
                                      test_allr),
                                     axis=2)
    # parse_args.batch_time * parse_args.data_size * (D + 2 + NUM_REGS) |->
    #                                       (parse_args.batch_time, parse_args.data_size, D + 2 + NUM_REGS)
    _, ravel_true_y_t_r_allr = ravel_pytree(true_y_t_r_allr)

    true_y0_t_r0_allr = np.concatenate((true_y0,
                                        np.expand_dims(
                                            np.repeat(t[0], parse_args.data_size), axis=1),
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
                                      np.ones((parse_args.batch_size, 1))),
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
                                      np.ones((parse_args.batch_size, 1))),
                                     axis=1)

        y0, y_n = sol_recursive(jet_wrap_predict(params), y, y_t[:, -1][0])

        none = np.zeros(parse_args.batch_size)
        r0 = np.sum(y ** 2, axis=1) ** 0.5
        r1 = np.sum(predictions_y ** 2, axis=1)
        r2 = np.sum(y_n[0] ** 2, axis=1)
        r3 = np.sum(y_n[1] ** 2, axis=1)
        r4 = np.sum(y_n[2] ** 2, axis=1)
        r5 = np.sum(y_n[3] ** 2, axis=1)
        r6 = np.sum(y_n[4] ** 2, axis=1)

        regs_map = {
            "none": none,
            "r0": r0,
            "r1": r1,
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
                                      np.ones((parse_args.data_size, 1))),
                                     axis=1)

        y0, y_n = sol_recursive(jet_wrap_predict(params), y, y_t[:, -1][0])

        none = np.zeros(parse_args.data_size)
        r0 = np.sum(y ** 2, axis=1) ** 0.5
        r1 = np.sum(predictions_y ** 2, axis=1)
        r2 = np.sum(y_n[0] ** 2, axis=1)
        r3 = np.sum(y_n[1] ** 2, axis=1)
        r4 = np.sum(y_n[2] ** 2, axis=1)
        r5 = np.sum(y_n[3] ** 2, axis=1)
        r6 = np.sum(y_n[4] ** 2, axis=1)

        regs_map = {
            "none": none,
            "r0": r0,
            "r1": r1,
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

    # unregularized system for counting NFE
    unreg_nodes_odeint = jax.jit(lambda y0, t, args: odeint(dynamics, y0, t, *args))
    unreg_nodes_odeint_vjp = jax.jit(lambda cotangent, y0, t, args:
                                     vjp_odeint(dynamics, y0, t, *args, nfe=True)[1](np.reshape(cotangent,
                                                                                                (parse_args.batch_time,
                                                                                                 parse_args.batch_size *
                                                                                                 (D + 1))))[-1])
    grad_loss_fn = grad(loss_fun)

    # full system for training
    nodes_odeint = build_odeint(reg_dynamics)
    grad_predict = jax.jit(grad(nodes_predict))

    # for testing
    nodes_odeint_test = build_odeint(test_reg_dynamics)

    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-3, gamma=0.99)
    opt_state = opt_init(fargs)

    assert parse_args.data_size % parse_args.batch_size == 0
    batch_per_epoch = parse_args.data_size // parse_args.batch_size

    for epoch in range(parse_args.nepochs):

        for batch in range(batch_per_epoch):
            itr = epoch * batch_per_epoch + batch + 1

            key, flat_batch_y0_t, flat_batch_y0_t_r0_allr0, batch_t, batch_y = pack_batch(rng)

            fargs = get_params(opt_state)

            # integrate unregularized system and count NFE
            pred_y_t, nfe = unreg_nodes_odeint(flat_batch_y0_t, batch_t, fargs)
            print("forward NFE: %d" % nfe)

            # integrate adjoint ODE to count NFE
            grad_loss = grad_loss_fn(ravel_batch_y_t(pred_y_t)[:, :, :D], batch_y)
            cotangent = np.concatenate((grad_loss,
                                        np.zeros((parse_args.batch_time, parse_args.batch_size, 1))),
                                       axis=2)
            nfe = unreg_nodes_odeint_vjp(cotangent, flat_batch_y0_t, batch_t, fargs)
            print("backward NFE: %d" % nfe)

            params_grad = np.array(grad_predict((batch_y, flat_batch_y0_t_r0_allr0, batch_t, *fargs))[3:])
            opt_state = opt_update(itr, params_grad, opt_state)

            if itr % parse_args.test_freq == 0:
                fargs = get_params(opt_state)

                pred_y_t_r_allr = ravel_true_y_t_r_allr(nodes_odeint_test(flat_true_y0_t_r0_allr, t, *fargs))

                pred_y = pred_y_t_r_allr[:, :, :D]
                pred_y_t_r = pred_y_t_r_allr[:, :, :D + 2]

                loss = loss_fun(pred_y, true_y)

                total_loss = total_loss_fun(pred_y_t_r, true_y)

                rk_reg = tuple(float(np.mean(pred_y_t_r_allr[1, :, reg_ind - NUM_REGS])) for reg_ind in range(NUM_REGS))
                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | ' + \
                            ' | '.join("%s {:.6f}" % reg for reg in REGS)

                print(print_str.format(itr, total_loss, loss, *rk_reg))
                print(print_str.format(itr, total_loss, loss, *rk_reg), file=sys.stderr)

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()


if __name__ == "__main__":
    assert os.path.exists(parse_args.dirname)
    run(parse_args.reg, parse_args.lam, random.PRNGKey(0), parse_args.dirname)
