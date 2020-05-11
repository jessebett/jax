"""
Toy example learning a sin wave with some separation.
"""
import argparse
import collections
import os
import pickle
import sys

import haiku as hk

import jax
from jax import lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet

from jax.scipy.special import expit as sigmoid

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1.4e-8)
parser.add_argument('--rtol', type=float, default=1.4e-8)
parser.add_argument('--method', type=str, default="dopri5")
parser.add_argument('--no_vmap', action="store_true")
parser.add_argument('--init_step', type=float, default=-1.)
parser.add_argument('--reg', type=str, choices=['none', 'r3'], default='none')
parser.add_argument('--test_freq', type=int, default=50)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resnet', action="store_true")
parser.add_argument('--no_count_nfe', action="store_true")
parser.add_argument('--num_blocks', type=int, default=6)
parse_args = parser.parse_args()


assert os.path.exists(parse_args.dirname)

# set up config

reg = parse_args.reg
lam = parse_args.lam
lam_w = parse_args.lam_w
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
dirname = parse_args.dirname
odenet = False if parse_args.resnet is True else True
count_nfe = False if parse_args.no_count_nfe or (not odenet) is True else True
vmap = False if parse_args.no_vmap is True else True
vmap = False
num_blocks = parse_args.num_blocks
ode_kwargs = {
    "atol": parse_args.atol,
    "rtol": parse_args.rtol,
    # "method": parse_args.method,
    # "init_step": parse_args.init_step
}


# some primitive functions
# def sigmoid(z):
#   """
#   Numerically stable sigmoid.
#   """
#   return 1/(1 + jnp.exp(-z))

act = lambda x: x


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  z_shape = z.shape
  z_t = jnp.concatenate((jnp.ravel(z), jnp.array([t])))

  def g(z_t):
    """
    Closure to expand z.
    """
    z, t = jnp.reshape(z_t[:-1], z_shape), z_t[-1]
    dz = jnp.ravel(f(z, t))
    dt = jnp.array([1.])
    dz_t = jnp.concatenate((dz, dt))
    return dz_t

  (y0, [y1h]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
  (y0, [y1, y2h]) = jet(g, (z_t, ), ((y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jet(g, (z_t, ), ((y0, y1, y2h), ))
  # (y0, [y1, y2, y3, y4h]) = jet(g, (z_t, ), ((y0, y1, y2, y3h), ))

  return jnp.reshape(y0[:-1], z_shape), [jnp.reshape(y1[:-1], z_shape), jnp.reshape(y0[:-1], z_shape)]
  # return y0[0], [y1[0], y2[0]]
  # return y0[0], [y1[0]]


class MLPDynamics(hk.Module):
    """
    NN_Dynamics for ODE as an MLP.
    """

    def __init__(self, dim):
        super(MLPDynamics, self).__init__()
        self.dim = dim
        self.hidden_dim = 100
        # self.lin1 = hk.Linear(self.hidden_dim)
        self.lin2 = hk.Linear(self.dim)

    def __call__(self, x, t):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, (-1, self.dim))

        out = act(x)
        # tt = jnp.ones_like(out[:, :1]) * t
        # t_out = jnp.concatenate([tt, out], axis=-1)
        # out = self.lin1(out)

        out = act(out)
        # tt = jnp.ones_like(out[:, :1]) * t
        # t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin2(out)

        return out / (1 + t)


def wrap_module(module, *module_args, **module_kwargs):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)
    return wrap


def initialization_data():
    """
    Data for initializing the modules.
    """
    data = jnp.zeros((3, 1)), 0.  # (batch_size, dimension)
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])
    ode_dim = 1

    initialization_data_ = initialization_data()

    dynamics = hk.transform(wrap_module(MLPDynamics, ode_dim))
    dynamics_params = dynamics.init(rng, *initialization_data_)
    dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)

    if reg:
        def reg_dynamics(y, t, params):
            """
            NN_Dynamics of regularization for ODE integration.
            """
            if reg == "none":
                return jnp.zeros(y.shape[0])
            else:
                # do r3 regularization
                y0, y_n = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
                r = y_n[-1]
                return jnp.sum(r ** 2, axis=[axis_ for axis_ in range(1, r.ndim)])

        def aug_dynamics(yr, t, params):
            """
            NN_Dynamics augmented with regularization.
            """
            y, r = yr
            dydt = dynamics_wrap(y, t, params)
            drdt = reg_dynamics(y, t, params)
            return dydt, drdt
        if vmap:
            # TODO: not sure if this is right, look at nodes_hk2.py
            nodeint = jax.vmap(lambda y0, t, params: odeint(aug_dynamics, aug_init(y0, 1), t, params, **ode_kwargs)[0],
                               (0, None, None), 1)
        else:
            nodeint = lambda y0, t, params: odeint(aug_dynamics, aug_init(y0), t, params, **ode_kwargs)[0]
    else:
        if vmap:
            nodeint = jax.vmap(lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[0],
                               (0, 0, None), 0)
        else:
            nodeint = lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[0]

    def ode(params, inputs):
        """
        Apply the ODE block.
        """
        out_ode, out_ode_r = nodeint(inputs, ts, params)
        return out_ode, out_ode_r

    def plot_ode(params, inputs, input_ts):
        """
        Apply the ODE block.
        """
        out_ode, out_ode_r = nodeint(inputs, input_ts, params)
        return out_ode, out_ode_r

    if count_nfe:
        if vmap:
            unreg_nodeint = jax.vmap(lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1],
                                     (0, 0, None))
        else:
            unreg_nodeint = lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1]

        @jax.jit
        def nfe_fn(params, inputs, targets):
            """
            Function to return NFE.
            """
            f_nfe = unreg_nodeint(inputs, ts, params)
            return jnp.mean(f_nfe)

    else:
        nfe_fn = None

    # return a dictionary of the three components of the model
    model = {
        "forward": ode,
        "params": dynamics_params,
        "nfe": nfe_fn,
        "plot_ode": plot_ode
    }

    return model


def aug_init(y, batch_size=-1):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    if batch_size == -1:
        batch_size = y.shape[0]
    return y, jnp.zeros(batch_size)


def _loss_fn(preds, targets):
    return jnp.mean((preds - targets) ** 2)


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, inputs, targets):
    """
    The loss function for training.
    """
    preds, regs = forward(params, inputs)
    loss_ = _loss_fn(preds[-1, :, 0], targets)
    reg_ = _reg_loss_fn(regs)
    weight_ = _weight_fn(params)
    return loss_ + lam * reg_ + lam_w * weight_


def init_data():
    """
    Initialize data.
    """
    num_train = 1000

    assert num_train % parse_args.batch_size == 0
    num_train_batches = num_train // parse_args.batch_size

    assert num_train % parse_args.test_batch_size == 0
    num_test_batches = num_train // parse_args.test_batch_size

    # make sure we always save the model on the last iteration
    assert num_train_batches * parse_args.nepochs % parse_args.save_freq == 0

    meta = {
        "num_batches": num_train_batches,
        "num_test_batches": num_test_batches
    }

    true_y0_range = 1
    true_fn = lambda x: 2 * x
    inputs = jnp.linspace(-true_y0_range, true_y0_range, num_train)[None].T
    targets = true_fn(inputs[:, 0])

    def gen_train_data(batch_size):
        """
        Generator for data.
        """
        key = rng
        num_batches = num_train // batch_size
        inds = jnp.arange(num_train)
        while True:
            key, subkey = jax.random.split(key)
            epoch_inds = jax.random.shuffle(subkey, inds)
            for i in range(num_batches):
                batch_inds = epoch_inds[i * batch_size: (i + 1) * batch_size]
                batch_inputs, batch_targets = inputs[batch_inds], targets[batch_inds]
                yield batch_inputs, batch_targets

    def gen_test_data(batch_size):
        """
        Generator for test data, so doesn't shuffle.
        """
        num_batches = num_train // batch_size
        inds = jnp.arange(num_train)
        while True:
            for i in range(num_batches):
                batch_inds = inds[i * batch_size: (i + 1) * batch_size]
                batch_inputs, batch_targets = inputs[batch_inds], targets[batch_inds]
                yield batch_inputs, batch_targets

    ds_train = gen_train_data(parse_args.batch_size)
    ds_train_eval = gen_test_data(parse_args.test_batch_size)

    return ds_train, ds_train_eval, meta


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    ds_train, ds_train_eval, meta = init_data()
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    model = init_model()
    forward = model["forward"]
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    def lr_schedule(train_itr):
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 60, 1e-1, id, 0,
                        lambda _: lax.cond(_epoch < 100, 1e-2, id, 0,
                                           lambda _: lax.cond(_epoch < 140, 1e-3, id, 1e-4, id)))

    # opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)
    # opt_init, opt_update, get_params = optimizers.momentum(step_size=1e-1, mass=0.9)
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
    # opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-1, gamma=0.99)
    opt_state = opt_init(model["params"])

    @jax.jit
    def update(_itr, _opt_state, _batch):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(_itr, grad_fn(get_params(_opt_state), *_batch), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        inputs, targets = _batch
        preds, regs = forward(params, inputs)
        loss_ = _loss_fn(preds[-1, :, 0], targets)
        reg_ = _reg_loss_fn(regs)
        total_loss_ = loss_ + lam * reg_
        return total_loss_, loss_, reg_

    def evaluate_loss(opt_state, ds_train_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_loss_aug_, sep_loss_, sep_loss_reg_, nfe = [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)

            test_batch_loss_aug_, test_batch_loss_, test_batch_loss_reg_ = sep_losses(opt_state, test_batch)

            if count_nfe:
                nfe.append(model["nfe"](get_params(opt_state), *test_batch))
            else:
                nfe.append(0)

            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_reg_.append(test_batch_loss_reg_)

        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_reg_ = jnp.array(sep_loss_reg_)
        nfe = jnp.array(nfe)

        return jnp.mean(sep_loss_aug_), jnp.mean(sep_loss_), jnp.mean(sep_loss_reg_), jnp.mean(nfe)

    loss_aug_, loss_, loss_reg_, nfe_ = evaluate_loss(opt_state, ds_train_eval)
    print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | ' \
                'Loss {:.6f} | r {:.6f} | NFE {:.6f}'.format(0, loss_aug_, loss_, loss_reg_, nfe_)
    print(print_str)
    inputs, _ = next(ds_train_eval)
    input_ts = jnp.linspace(0., 1., num=100)
    out, _ = model["plot_ode"](get_params(opt_state), inputs, input_ts)
    fig, ax = plt.subplots()

    for ex_num in range(10):
        ax.plot(input_ts, out[:, ex_num * 100, 0], c="blue")
    plt.savefig("{:08d}.png".format(0))
    plt.clf()
    plt.close(fig)

    itr = 0
    info = collections.defaultdict(dict)
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)

            itr += 1

            opt_state = update(itr, opt_state, batch)

            if itr % parse_args.test_freq == 0:
                with jax.disable_jit():
                    loss_aug_, loss_, loss_reg_, nfe_ = evaluate_loss(opt_state, ds_train_eval)

                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f} | NFE {:.6f}'.format(itr, loss_aug_, loss_, loss_reg_, nfe_)

                print(print_str)

                inputs, _ = next(ds_train_eval)
                input_ts = jnp.linspace(0., 1., num=100)
                out, _ = model["plot_ode"](get_params(opt_state), inputs, input_ts)
                fig, ax = plt.subplots()

                for ex_num in range(10):
                    ax.plot(input_ts, out[:, ex_num * 100, 0], c="blue")
                plt.savefig("{:08d}.png".format(itr))
                plt.clf()
                plt.close(fig)

                outfile = open("%s/reg_%s_lam_%.4e_num_blocks_%d_info.txt" % (dirname, reg, lam, num_blocks), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["loss_aug"] = loss_aug_
                info[itr]["loss"] = loss_
                info[itr]["loss_reg"] = loss_reg_
                info[itr]["nfe"] = nfe_

            if itr % parse_args.save_freq == 0:
                if odenet:
                    param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                else:
                    param_filename = "%s/reg_%s_lam_%.4e_num_blocks_%d_%d_fargs.pickle" % (dirname, reg, lam, num_blocks, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()

            outfile = open("%s/reg_%s_lam_%.4e_num_blocks_%d_iter.txt" % (dirname, reg, lam, num_blocks), "a")
            outfile.write("Iter: {:04d}\n".format(itr))
            outfile.close()
    meta = {
        "info": info,
        "args": parse_args
    }
    outfile = open("%s/reg_%s_lam_%.4e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "wb")
    pickle.dump(meta, outfile)
    outfile.close()


if __name__ == "__main__":
    run()
