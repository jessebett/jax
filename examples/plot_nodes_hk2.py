"""
For plotting jobs from nodes_hk2.py
"""

import pickle
from glob import glob
from functools import reduce, partial
import argparse

import jax
from jax.experimental.ode import _heun_odeint, _fehlberg_odeint, _bosh_odeint, _owrenzen_odeint, _dopri5_odeint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as onp

import jax.numpy as jnp

from nodes_hk2 import init_model, init_data, _reg_loss_fn, _loss_fn

parser = argparse.ArgumentParser('Plot')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=640)
parser.add_argument('--save_freq', type=int, default=640)
parser.add_argument('--data_root', type=str, default="./")  # TODO: hard code this on cluster
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--method', type=str,
                    choices=["heun", "fehlberg", "bosh", "owrenzen", "dopri"],
                    default="dopri")  # TODO: flag that needs to be set!!
parser.add_argument('--reg', type=str, choices=['none', 'r2', 'r3', 'r4', 'r5'], default='none')
parser.add_argument('--reg_result', type=str, choices=['none', 'r2', 'r3', 'r4', 'r5'], default='none')
parser.add_argument('--dirname', type=str, default='tmp')
parse_args = parser.parse_args()

# TODO: set this to absolute path
dirname = parse_args.dirname
reg = parse_args.reg
reg_result = parse_args.reg_result
num_blocks = 0
ode_kwargs = {
    "atol": 1.4e-8,
    "rtol": 1.4e-8,
    # "method": parse_args.method,
    # "init_step": parse_args.init_step
}
methods = {"heun": _heun_odeint,
           "fehlberg": _fehlberg_odeint,
           "bosh": _bosh_odeint,
           "owrenzen": _owrenzen_odeint,
           "dopri": _dopri5_odeint
           }
num_stages = {"heun": 1,
              "fehlberg": 2,
              "bosh": 3,
              "owrenzen": 5,
              "dopri": 6
              }

forward, model = init_model()
forward = jax.jit(forward)
_loss_fn = jax.jit(_loss_fn)
method = parse_args.method
count_nfe = jax.jit(partial(model["plot_nfe"], methods[method]))
_, ds_train_eval, meta = init_data()
num_test_batches = meta["num_test_batches"]


def parse_lam(filename):
    """
    Parse lambda from a filename.
    """
    return float(filename.split("_")[3])


lams = list(map(parse_lam, glob("%s/*%s*meta.pickle" % (dirname, reg))))


def sanity_check(lam):
    """
    Sanity check that test and train loss are diff.
    """
    meta_file = open("%s/reg_%s_lam_%.18e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta = pickle.load(meta_file)

    itr = 96000
    nfe_test_filename = "%s/reg_test_%s_lam_%.12e_%d_nfe.pickle" % (dirname, reg, lam, itr)
    nfe_test_file = open(nfe_test_filename, "rb")
    nfe_test = pickle.load(nfe_test_file)
    nfe_test_file.close()

    nfe_filename = "%s/reg_%s_lam_%.12e_%d_nfe.pickle" % (dirname, reg, lam, itr)
    nfe_file = open(nfe_filename, "rb")
    nfe = pickle.load(nfe_file)
    nfe_file.close()

    print()

    return nfe, loss


def get_info(reg, dirname, method, lam):
    """
    Get (final NFE, final loss) pair for a given lambda.
    """
    meta_file = open("%s/reg_%s_lam_%.18e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta = pickle.load(meta_file)
    meta_file.close()

    itr = 96000
    # TODO: train or test (make this a separate function so you don't fuck up)
    # TODO: you'll have to make this backwards compatible for dopri5
    nfe_filename = "%s/reg_%s_lam_%.12e_%d_%s_nfe.pickle" % (dirname, reg, lam, itr, method)
    try:
        nfe_file = open(nfe_filename, "rb")
        nfe = pickle.load(nfe_file)
        nfe_file.close()
    except IOError:
        print("Calculating NFE for %.4e" % lam)
        param_file = open("%s/reg_%s_lam_%.18e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
        params = pickle.load(param_file)
        nfes = []
        for test_batch_num in range(num_test_batches):
            print(test_batch_num, num_test_batches)
            test_batch = next(ds_train_eval)
            nfes.append(count_nfe(params, *test_batch))
        nfe = onp.mean(nfes)
        nfe_file = open(nfe_filename, "wb")
        pickle.dump(nfe, nfe_file)
        nfe_file.close()

    # loss = 1 - meta["info"][itr]["acc"]
    # log-log and log-linear both look good
    nfe = nfe
    # reg_val = meta["info"][itr]["loss_reg"]
    reg_val = meta["info"][itr]["loss_reg"]

    # TODO: train or test
    loss = meta["info"][itr]["loss"]
    # test_loss_filename = "%s/test_loss_%s_lam_%.18e.pickle" % (dirname, reg, lam)
    # test_loss_file = open(test_loss_filename, "rb")
    # test_loss = pickle.load(test_loss_file)
    # test_loss_file.close()

    # TODO: train or test
    return nfe, loss


def pareto_plot_nfe():
    """
    Create pareto plot.
    """
    cm = plt.get_cmap('viridis')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [30, 1], "wspace": 0.05})

    sorted_lams = sorted(lams)
    x, y = zip(*map(get_info, sorted_lams))
    anno = sorted_lams

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
    ax.set_ylabel("Unregularized Test Loss")

    norm = mpl.colors.LogNorm(vmin=anno[0], vmax=anno[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_label(r'Regularization Weight ($\lambda$)')

    plt.gcf().subplots_adjust(right=0.88, left=0.13)
    # plt.gcf().subplots_adjust(right=0.88)

    plt.savefig("%s/loss_nfe_test_%s_pareto.pdf" % (dirname, reg))
    plt.savefig("%s/loss_nfe_test_%s_pareto.png" % (dirname, reg))
    plt.clf()
    plt.close(fig)


def nfe_train_test():
    """
    Create pareto plot.
    """
    cm = plt.get_cmap('viridis')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [30, 1], "wspace": 0.05})

    itr = 96000

    def get_nfe_train_test(lam):
        nfe_filename = "%s/reg_test_%s_lam_%.12e_%d_nfe.pickle" % (dirname, reg, lam, itr)
        nfe_file = open(nfe_filename, "rb")
        nfe_test = pickle.load(nfe_file)
        nfe_file.close()

        nfe_filename = "%s/reg_%s_lam_%.12e_%d_nfe.pickle" % (dirname, reg, lam, itr)
        nfe_file = open(nfe_filename, "rb")
        nfe = pickle.load(nfe_file)
        nfe_file.close()

        return nfe, nfe_test

    sorted_lams = sorted(lams)
    nfe_train, nfe_test = zip(*map(get_nfe_train_test, sorted_lams))
    anno = sorted_lams

    num_points = len(nfe_train)
    c_spacing = onp.linspace(0, 1, num=num_points)
    cmap = lambda ind: cm(c_spacing[ind])

    for i in range(num_points):
        ax.scatter(nfe_train[i], nfe_test[i], c=cmap(i))

    ax.set_xlabel("Final Average NFE on Train")
    ax.set_ylabel("Final Average NFE on Test")

    norm = mpl.colors.LogNorm(vmin=anno[0], vmax=anno[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_label(r'Regularization Weight ($\lambda$)')

    plt.gcf().subplots_adjust(right=0.88, left=0.13)
    # plt.gcf().subplots_adjust(right=0.88)

    plt.savefig("%s/nfe_train_test_%s_pareto.pdf" % (dirname, reg))
    plt.savefig("%s/nfe_train_test_%s_pareto.png" % (dirname, reg))
    plt.clf()
    plt.close(fig)


def pareto_plot_nfe_all_orders():
    """
    Create pareto plot.
    """
    cm = plt.get_cmap('viridis')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [30, 1], "wspace": 0.05})

    sorted_lams = sorted(lams)

    reg_dirs = {
        "r2": "2020-05-02-22-08-30",
        "r3": "2020-05-01-12-02-58",
        # "r4": "2020-05-10-10-14-19",
        # "r5": "2020-05-10-21-22-28"
    }

    regs = ["r2", "r3"]
    regs_opacity = onp.linspace(0.1, 0.9, num=len(regs))

    anno = sorted_lams

    for reg, dir in reg_dirs.items():
        x, y = zip(*map(partial(get_info, reg, dir), sorted_lams))

        num_points = len(x)
        c_spacing = onp.linspace(0, 1, num=num_points)
        cmap = lambda ind: cm(c_spacing[ind])

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
        ax.plot(pf_X, pf_Y, c=str(regs_opacity[regs.index(reg)]))

    ax.set_xlabel("Average Number of Function Evaluations")
    ax.set_ylabel("Unregularized Training Loss")

    # ax.set_ylim(bottom=3e-3, top=3.6e-3)

    norm = mpl.colors.LogNorm(vmin=anno[0], vmax=anno[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_label(r'Regularization Weight ($\lambda$)')

    plt.gcf().subplots_adjust(right=0.88, left=0.14)
    # plt.gcf().subplots_adjust(right=0.88)

    plt.savefig("%s/all_nfe_loss_pareto.pdf" % dirname)
    plt.savefig("%s/all_nfe_loss_pareto.png" % dirname)
    plt.clf()
    plt.close(fig)


def get_nfe_dist(lam):
    """
    Return a dict mapping itr -> NFE for example in batch
    """
    meta_file = open("%s/reg_%s_lam_%.4e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta = pickle.load(meta_file)

    nfes_filename = "%s/reg_%s_lam_%.12e_nfe_dist.pickle" % (dirname, reg, lam)
    try:
        nfes_file = open(nfes_filename, "rb")
        nfes = pickle.load(nfes_file)
        nfes_file.close()
    except IOError:
        print("Calculating NFE dist for %.8e" % lam)

        def _get_nfe_dist(itr):
            param_file = open("%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
            params = pickle.load(param_file)
            nfes_itr = []
            for test_batch_num in range(num_test_batches):
                test_batch = next(ds_train_eval)
                stuff = model["nfe"](params, *test_batch)
                nfes_itr.extend(stuff)
            outfile = open("%s/iter.txt" % dirname, "a")
            outfile.write("Lam {:.12e} | Iter: {:05d}\n".format(lam, itr))
            outfile.close()
            return nfes_itr
        nfes = dict((itr, _get_nfe_dist(itr)) for itr in meta["info"])
        nfe_file = open(nfes_filename, "wb")
        pickle.dump(nfes, nfe_file)
        nfe_file.close()

    meta_file.close()
    return nfes


def histogram_nfe():
    """
    For each lam, plot a histogram of NFE
    """
    sorted_lams = sorted(lams)[:-41]
    all_nfes = list(map(get_nfe_dist, sorted_lams))

    for (ind, lam), nfes in zip(enumerate(sorted_lams), all_nfes):
        print(ind)
        fig, ax = plt.subplots()
        nfes_comb = reduce(lambda i, j: i + j, nfes.values())
        nfes_comb = (onp.asarray(nfes_comb) - 1) / 6
        ax.hist(nfes_comb, bins=[3, 4, 5], density=True)
        plt.title("Lam: %.8e" % lam)
        figname = "{}/{:04d}.png".format(dirname, ind)
        plt.savefig(figname)
        plt.clf()
        plt.close(fig)


def get_info_r(lam):
    """
    Get (NFE, reg) pairs for a lam.
    """
    meta_file = open("%s/reg_%s_lam_%.4e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta = pickle.load(meta_file)
    meta_file.close()
    info = meta["info"]

    return [(onp.log10(info[itr]["loss_reg"]), onp.log10(info[itr]["nfe"])) for itr in info]


def nfe_r():
    """
    Correlation between regularization and NFE.
    """
    sorted_lams = sorted(lams)[:-41]
    lam_pts = list(map(get_info_r, sorted_lams))

    fig, ax = plt.subplots()

    cm = plt.get_cmap('viridis')

    num_points = len(sorted_lams)
    c_spacing = onp.linspace(0, 1, num=num_points)
    cmap = lambda ind: cm(c_spacing[ind])

    for i, lam_pt in enumerate(lam_pts):
        x, y = zip(*lam_pt)
        ax.scatter(x, y, c=cmap(i))

    ax.set_xlabel("Mean Regularization")
    ax.set_ylabel("Average Number of Function Evaluations")

    figname = "%s/scatter.png" % dirname
    plt.savefig(figname)
    plt.clf()
    plt.close(fig)


def get_r(lam, reg_order):
    """
    Get regularization of a particular order.
    """
    meta_file = open("%s/reg_%s_lam_%.4e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta = pickle.load(meta_file)

    reg_filename = "%s/reg_%s_lam_%.12e_%s.pickle" % (dirname, reg, lam, reg_order)
    try:
        print("Loading %s dist %.8e" % (reg_order, lam))
        reg_file = open(reg_filename, "rb")
        regs = pickle.load(reg_file)
        reg_file.close()
    except IOError:
        print("Calculating %s dist %.8e" % (reg_order, lam))

        def _get_reg(itr):
            param_file = open("%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
            params = pickle.load(param_file)
            regs_itr = []
            for test_batch_num in range(num_test_batches):
                test_batch = next(ds_train_eval)
                _, regs = forward(params, test_batch[0])
                regs_itr.extend(regs)
            outfile = open("%s/iter.txt" % dirname, "a")
            outfile.write("Lam {:.12e} | Iter: {:05d}\n".format(lam, itr))
            outfile.close()
            return regs_itr
        # regs = dict((itr, _get_reg(itr)) for itr in meta["info"])
        if reg_order == "r3":
            regs = dict((itr, meta["info"][itr]["loss_reg"]) for itr in meta["info"])
        else:
            regs = dict((itr, _get_reg(itr)) for itr in meta["info"])
        reg_file = open(reg_filename, "wb")
        pickle.dump(regs, reg_file)
        reg_file.close()

    meta_file.close()
    for itr in regs:
        regs[itr] = onp.mean(regs[itr])
    return regs


def get_test_loss(lam):
    """
    Get regularization of a particular order.
    """
    test_loss_filename = "%s/test_loss_%s_lam_%.18e.pickle" % (dirname, reg, lam)
    try:
        # print("Loading test loss %.18e" % lam)
        test_loss_file = open(test_loss_filename, "rb")
        test_loss = pickle.load(test_loss_file)
        test_loss_file.close()
    except IOError:
        print("Calculating test loss %.18e" % lam)

        itr = 96000
        param_file = open("%s/reg_%s_lam_%.18e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
        params = pickle.load(param_file)
        losses = onp.zeros(num_test_batches)
        for test_batch_num in range(num_test_batches):
            print(test_batch_num)
            test_batch = next(ds_train_eval)
            images, labels = test_batch
            logits, regs = forward(params, images)
            loss_ = _loss_fn(logits, labels)
            losses[test_batch_num] = loss_
        test_loss = onp.mean(losses)
        test_loss_file = open(test_loss_filename, "wb")
        pickle.dump(test_loss, test_loss_file)
        test_loss_file.close()
    return test_loss


def pareto_nfe_r():
    """
    Pareto of NFE against different orders of regularization.
    """
    reg_orders = ["r0", "r1", "r2", "r3", "r4", "r5", "r6"]
    reg_orders = ["r3", "r4"]
    for lam_rank, lam in enumerate(sorted(lams)):
        for reg_order in reg_orders:
            get_r(lam, reg_order)


def r_corr():
    """
    Are the regularizations correlated.
    """
    sorted_lams = sorted(lams)
    try:
        reg_file = open("%s/r3_corr.pickle" % dirname, "rb")
        r3_regs = pickle.load(reg_file)
        reg_file.close()

        reg_file = open("%s/r4_corr.pickle" % dirname, "rb")
        r4_regs = pickle.load(reg_file)
        reg_file.close()
    except IOError:
        r3_regs = list(map(lambda lam: get_r(lam, "r3"), sorted_lams))
        r4_regs = list(map(lambda lam: get_r(lam, "r4"), sorted_lams))

        reg_file = open("%s/r3_corr.pickle" % dirname, "wb")
        pickle.dump(r3_regs, reg_file)
        reg_file.close()

        reg_file = open("%s/r4_corr.pickle" % dirname, "wb")
        pickle.dump(r4_regs, reg_file)
        reg_file.close()

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [30, 1], "wspace": 0.05})

    cm = plt.get_cmap('viridis')

    num_points = len(sorted_lams)
    c_spacing = onp.linspace(0, 1, num=num_points)
    cmap = lambda ind: cm(c_spacing[ind])

    for i, pt in enumerate(zip(r3_regs, r4_regs)):
        x, y = pt
        ax.scatter(x, y, c=cmap(i))

    ax.set_xlabel("$\mathcal{R}_3$")
    ax.set_ylabel("$\mathcal{R}_4$")

    # ax.set_xlabel("r3")
    # ax.set_ylabel("r4")

    norm = mpl.colors.LogNorm(vmin=sorted_lams[0], vmax=sorted_lams[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_label(r'Regularization Weight ($\lambda$)')

    plt.gcf().subplots_adjust(right=0.88, left=0.13)

    plt.savefig("%s/r_corr.pdf" % dirname)
    plt.savefig("%s/r_corr.png" % dirname)
    plt.clf()
    plt.close(fig)


def r_over_training():
    """
    Track regularization over training.
    """
    sorted_lams = sorted(lams)
    try:
        reg_file = open("%s/r3_training.pickle" % dirname, "rb")
        r3_regs = pickle.load(reg_file)
        reg_file.close()

        reg_file = open("%s/r4_training.pickle" % dirname, "rb")
        r4_regs = pickle.load(reg_file)
        reg_file.close()
    except IOError:
        r3_regs = list(map(lambda lam: get_r(lam, "r3"), sorted_lams))
        r4_regs = list(map(lambda lam: get_r(lam, "r4"), sorted_lams))

        reg_file = open("%s/r3_training.pickle" % dirname, "wb")
        pickle.dump(r3_regs, reg_file)
        reg_file.close()

        reg_file = open("%s/r4_training.pickle" % dirname, "wb")
        pickle.dump(r4_regs, reg_file)
        reg_file.close()

    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 14}
    # plt.rc('font', **font)
    # plt.rc('text', usetex=True)
    # fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [30, 1], "wspace": 0.05})
    #
    # cm = plt.get_cmap('viridis')
    #
    # num_points = len(sorted_lams)
    # c_spacing = onp.linspace(0, 1, num=num_points)
    # cmap = lambda ind: cm(c_spacing[ind])

    fig, ax_r3 = plt.subplots()
    ax_r3.set_xlabel("Training Iteration")
    ax_r3.set_ylabel("Mean $\mathcal{R}_3$")
    ax_r3.tick_params(axis='y', labelcolor="red")
    ax_r4 = ax_r3.twinx()
    ax_r4.set_ylabel("Mean $\mathcal{R}_4$")
    ax_r4.tick_params(axis='y', labelcolor="blue")

    itrs = list(r3_regs[0].keys())

    for r3_reg, r4_reg in zip(r3_regs, r4_regs):
        ax_r3.plot(itrs, list(r3_reg.values()), color="red")
        ax_r4.plot(itrs, list(r4_reg.values()), color="blue")

    fig.tight_layout()
    # ax.set_xlabel("r3")
    # ax.set_ylabel("r4")

    # norm = mpl.colors.LogNorm(vmin=sorted_lams[0], vmax=sorted_lams[-1])
    # cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    # cb1.set_label(r'Regularization Weight ($\lambda$)')
    #
    # plt.gcf().subplots_adjust(right=0.88, left=0.13)

    # plt.savefig("%s/r_corr.pdf" % dirname)
    plt.savefig("%s/r_train.png" % dirname)
    plt.clf()
    plt.close(fig)


def plot_nfe_over_training():
    cm = plt.get_cmap('viridis')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots()

    lam = 1e3
    meta_reg_file = open("%s/reg_%s_lam_%.18e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta_reg = pickle.load(meta_reg_file)

    meta_none_file = open("%s/reg_none_lam_%.18e_num_blocks_%d_meta.pickle" % (dirname, 0, 1), "rb")
    meta_none = pickle.load(meta_none_file)

    itrs = list(meta_reg["info"].keys())
    reg_nfe = [meta_reg["info"][itr]["nfe"] for itr in itrs]
    none_nfe = [meta_none["info"][itr]["nfe"] for itr in itrs]

    ax.plot(itrs, reg_nfe, c="blue", label="regularized")
    ax.plot(itrs, none_nfe, c='red', label="unregularized")

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Number of Function Evaluations")

    plt.legend()
    plt.savefig("%s/nfe_over_training.pdf" % dirname)
    plt.savefig("%s/nfe_over_training.png" % dirname)
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    get_info(reg, dirname, method, parse_args.lam)

    # for ind, lam in enumerate(lams):
    #     print(ind, len(lams))
    #     get_test_loss(lam)

    # nfe_train_test()
    # plot_nfe_over_training()
    # pareto_plot_nfe_all_orders()

    # nfe_r()
    # pareto_plot_nfe()
    # histogram_nfe()
    # pareto_nfe_r()
    # r_corr()
    # r_over_training()
