"""
For plotting jobs from nodes_hk2.py
"""

import pickle
from glob import glob
from functools import reduce

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as onp

from nodes_hk2 import init_model, init_data, _reg_loss_fn

# TODO: set this to absolute path
dirname = "2020-04-24-15-59-17"
reg = "r3"
num_blocks = 0

forward, model = init_model()
_, ds_train_eval, meta = init_data()
num_test_batches = meta["num_test_batches"]


def parse_lam(filename):
    """
    Parse lambda from a filename.
    """
    return float(filename.split("_")[3])


lams = list(map(parse_lam, glob("%s/*r3*meta.pickle" % dirname)))


def get_info(lam):
    """
    Get (final NFE, final loss) pair for a given lambda.
    """
    meta_file = open("%s/reg_%s_lam_%.4e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "rb")
    meta = pickle.load(meta_file)

    itr = 96000
    nfe_filename = "%s/reg_%s_lam_%.4e_%d_nfe.pickle" % (dirname, reg, lam, itr)
    try:
        nfe_file = open(nfe_filename, "rb")
        nfe = pickle.load(nfe_file)
        nfe_file.close()
    except IOError:
        print("Calculating NFE for %.4e" % lam)
        param_file = open("%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr), "rb")
        params = pickle.load(param_file)
        nfes = []
        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)
            nfes.append(model["nfe"](params, *test_batch))
        nfe = onp.mean(nfes)
        nfe_file = open(nfe_filename, "wb")
        pickle.dump(nfe, nfe_file)
        nfe_file.close()

    # loss = 1 - meta["info"][itr]["acc"]
    # log-log and log-linear both look good
    # nfe = onp.log10(nfe)
    reg_val = meta["info"][itr]["loss_reg"]
    loss = meta["info"][itr]["loss"]

    meta_file.close()
    return onp.log10(reg_val), onp.log10(loss)


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

    sorted_lams = sorted(lams)[:-41]
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
    pareto_front = sorted_list[1:9] + [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    # ax.plot(pf_X, pf_Y, c='0.35')

    ax.set_xlabel("Log Mean Regularization")
    ax.set_ylabel("Log Unregularized Training Loss")

    norm = mpl.colors.LogNorm(vmin=anno[0], vmax=anno[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax_leg, cmap=cm, norm=norm, orientation='vertical')
    cb1.set_label(r'Regularization Weight ($\lambda$)')

    plt.gcf().subplots_adjust(right=0.88, left=0.13)
    # plt.gcf().subplots_adjust(right=0.88)

    plt.savefig("%s/loss_reg_pareto.pdf" % dirname)
    plt.savefig("%s/loss_reg_pareto.png" % dirname)
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
    return regs


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
    # r3_regs = list(map(lambda lam: get_r(lam, "r3")[96000], sorted(lams)))
    r4_regs = list(map(lambda lam: get_r(lam, "r4")[96000], sorted(lams)))

    # fig, ax = plt.subplots()
    # ax.scatter(r3_regs, r4_regs)
    # ax.set_xlabel("r3")
    # ax.set_ylabel("r4")
    #
    # figname = "%s/r_corr.png" % dirname
    # plt.savefig(figname)
    # plt.clf()
    # plt.close(fig)


if __name__ == "__main__":
    # nfe_r()
    # pareto_plot_nfe()
    # histogram_nfe()
    # pareto_nfe_r()
    r_corr()
