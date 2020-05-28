"""
For plotting jobs from nodes_tabular.py
"""

import pickle
from glob import glob
from functools import reduce, partial
import argparse
import operator as op

import jax
from jax.experimental.ode import _heun_odeint, _fehlberg_odeint, _bosh_odeint, _owrenzen_odeint, _dopri5_odeint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as onp

import jax.numpy as jnp

from nodes_tabular import init_model, init_data, _loss_fn

parser = argparse.ArgumentParser('Plot')
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none', 'r2', 'r3', 'r4', 'r5'], default='none')
parser.add_argument('--dirname', type=str, default='tmp')
parse_args = parser.parse_args()

reg = parse_args.reg
dirname = parse_args.dirname
lam = parse_args.lam

forward, model = init_model(43)  # how do you sleep at night
forward_exact = jax.jit(model["forward_exact"])
ds_train, ds_test_eval, meta = init_data()
num_batches = meta["num_batches"]


def parse_lam(filename):
    """
    Parse lambda from a filename.
    """
    return float(filename.split("_")[3])


# lams = list(map(parse_lam, glob("%s/*%s*meta.pickle" % (dirname, reg))))


def get_nfe(reg, dirname, lam):
    """
    Get vmapped NFE for a model.
    """
    itr = 15000
    nfe_filename = "%s/reg_%s_lam_%.18e_%d_nfe.pickle" % (dirname, reg, lam, itr)
    try:
        nfe_file = open(nfe_filename, "rb")
        nfe = pickle.load(nfe_file)
        nfe_file.close()
    except IOError:
        print("Calculating NFE for %.4e" % lam)
        param_file = open("%s/reg_%s_lam_%.18e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
        params = pickle.load(param_file)
        nfe, bs = [], []
        _key = jax.random.PRNGKey(0)
        for test_batch_num in range(num_batches):
            _key, = jax.random.split(_key, num=1)
            batch = next(ds_train)

            nfe.append(model["plot_nfe"](_key, params, batch))
            bs.append(len(batch))

        nfe = jnp.array(nfe)
        bs = jnp.array(bs)

        nfe = onp.average(nfe, weights=bs)
        nfe_file = open(nfe_filename, "wb")
        pickle.dump(nfe, nfe_file)
        nfe_file.close()

    return nfe


def get_exact_likelihood(reg, dirname, lam):
    """
    Get exact log-likelihood using brute force trace computation.
    """
    itr = 15000
    loss_filename = "%s/reg_%s_lam_%.18e_%d_loss.pickle" % (dirname, reg, lam, itr)
    try:
        loss_file = open(loss_filename, "rb")
        loss = pickle.load(loss_file)
        loss_file.close()
    except IOError:
        print("Calculating Loss for %.4e" % lam)
        param_file = open("%s/reg_%s_lam_%.18e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
        params = pickle.load(param_file)
        loss, bs = [], []
        _key = jax.random.PRNGKey(0)
        for batch_num in range(num_batches):
            print(batch_num, num_batches)
            _key, = jax.random.split(_key, num=1)
            batch = next(ds_train)

            z, delta_logp, _ = forward_exact(_key, params, batch)
            loss_ = _loss_fn(z, delta_logp)
            loss.append(loss_)
            bs.append(len(batch))

        loss = jnp.array(loss)
        bs = jnp.array(bs)

        loss = onp.average(loss, weights=bs)
        loss_file = open(loss_filename, "wb")
        pickle.dump(loss, loss_file)
        loss_file.close()

    return loss


def pareto_plot_nfe():
    """
    Create pareto plot.
    """
    # TODO: check for nans, remove other outliers
    cm = plt.get_cmap('viridis')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [30, 1], "wspace": 0.05})

    sorted_lams = sorted(lams)
    x, y = zip(*map(partial(get_info, reg, dirname, method), sorted_lams))
    x = onp.array(x)
    y = onp.array(y)
    anno = onp.array(sorted_lams)

    # filter out nans and corresponding lams
    finite_mask = onp.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    anno = anno[finite_mask]

    num_points = len(x)
    c_spacing = onp.linspace(0, 1, num=num_points)
    cmap = lambda ind: cm(c_spacing[ind])

    for i in range(num_points):
        ax.scatter(x[i], y[i], c=cmap(i))

    # plot the pareto front
    maxY = False
    sorted_list = sorted([[x[i], y[i]] for i in range(len(x))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    ax.plot(pf_X, pf_Y, c='0.35')

    ax.set_xlabel("Average Number of Function Evaluations")
    ax.set_ylabel("Unregularized Training Loss")

    norm = mpl.colors.LogNorm(vmin=anno[0], vmax=anno[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_label(r'Regularization Weight ($\lambda$)')

    plt.gcf().subplots_adjust(right=0.88, left=0.13)
    # plt.gcf().subplots_adjust(right=0.88)

    plt.savefig("%s/loss_nfe_%s_pareto.pdf" % (dirname, reg))
    plt.savefig("%s/loss_nfe_%s_pareto.png" % (dirname, reg))
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    print(get_exact_likelihood(reg, dirname, lam))
