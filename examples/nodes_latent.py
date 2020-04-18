"""
Implementing Latent ODE!! YEET
"""
import argparse
import collections
import os
import pickle
import sys

import haiku as hk

from examples.timeseries import Periodic1D

import jax
from jax import lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=200)
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--subsample', type=int, default=50)
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
num_blocks = parse_args.num_blocks
ode_kwargs = {
    "atol": parse_args.atol,
    "rtol": parse_args.rtol,
    "method": parse_args.method,
    "init_step": parse_args.init_step
}


# some primitive functions
def sigmoid(z):
  """
  Numerically stable sigmoid.
  """
  return 1/(1 + jnp.exp(-z))


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  def g(z):
    """
    Closure to expand z.
    """
    return f(z, t)

  (y0, [y1h]) = jet(g, (z, ), ((jnp.ones_like(z), ), ))
  (y0, [y1, y2h]) = jet(g, (z, ), ((y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jet(g, (z, ), ((y0, y1, y2h), ))

  return (y0, [y1, y2])


class LatentGRU(hk.Module):
    """
    Modified GRU unit to deal with latent state.
    """
    def __init__(self,
                 latent_dim,
                 n_units,
                 **init_kwargs):
        super(LatentGRU, self).__init__()
        self.latent_dim = latent_dim

        self.update_gate = hk.Sequential([
           hk.Linear(n_units, **init_kwargs),
           jnp.tanh,
           hk.Linear(latent_dim, **init_kwargs),
           sigmoid
        ])

        self.reset_gate = hk.Sequential([
           hk.Linear(n_units, **init_kwargs),
           jnp.tanh,
           hk.Linear(latent_dim, **init_kwargs),
           sigmoid
        ])

        self.new_state_net = hk.Sequential([
           hk.Linear(n_units, **init_kwargs),
           jnp.tanh,
           hk.Linear(latent_dim * 2, **init_kwargs)
        ])

    def __call__(self, y_mean, y_std, x):
        y_concat = jnp.concatenate([y_mean, y_std, x], axis=-1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = jnp.concatenate([y_mean * reset_gate, y_std * reset_gate, x], axis=-1)

        new_state = self.new_state_net(concat)
        new_state_mean, new_state_std = new_state[..., :self.latent_dim], new_state[..., self.latent_dim:]
        new_state_std = jnp.abs(new_state_std)

        new_y_mean = (1-update_gate) * new_state_mean + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        new_y_std = jnp.abs(new_y_std)
        return new_y_mean, new_y_std


class Dynamics(hk.Module):
    """
    ODE-RNN dynamics.
    """

    def __init__(self,
                 latent_dim,
                 layers,
                 units):
        super(Dynamics, self).__init__()
        self.latent_dim = latent_dim
        self.model = hk.Sequential([unit for _ in range(layers + 1) for unit in
                                    [jnp.tanh, hk.Linear(units)]] +
                                   [jnp.tanh, hk.Linear(latent_dim)]
                                   )

    def __call__(self, y, t):
        y = jnp.reshape(y, (-1, self.latent_dim))
        y_t = jnp.concatenate((y, jnp.ones((y.shape[0], 1)) * t), axis=1)
        return self.model(y_t)


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


def initialization_data(rec_dim, gen_dim, data_dim):
    """
    Creates data for initializing each of the modules based on the shapes of init_data.
    """
    data = {
        "gru": (jnp.zeros(rec_dim), jnp.zeros(rec_dim), jnp.zeros((data_dim, ))),
        "rec_dynamics": (jnp.zeros(rec_dim), 0.),
        "rec_to_gen": (jnp.zeros(rec_dim), jnp.zeros(rec_dim)),
        "gen_dynamics": (jnp.zeros(gen_dim), 0.),
        "gen_to_data": jnp.zeros(gen_dim)
    }
    return data


def init_model(rec_ode_kwargs,
               gen_ode_kwargs,
               rec_dim=20,
               gen_dim=10,
               data_dim=1,
               rec_layers=1,
               gen_layers=1,
               dynamics_units=100,
               gru_units=100,):
    """
    Instantiates transformed submodules of model and their parameters.
    """

    initialization_data_ = initialization_data(rec_dim,
                                               gen_dim,
                                               data_dim)

    init_kwargs = {
        "w_init": hk.initializers.RandomNormal(mean=0, stddev=0.1),
        "b_init": jnp.zeros
    }

    gru = hk.transform(wrap_module(LatentGRU,
                                   latent_dim=rec_dim,
                                   n_units=gru_units,
                                   **init_kwargs))
    gru_params = gru.init(rng, *initialization_data_["gru"])

    rec_dynamics = hk.transform(wrap_module(Dynamics,
                                            latent_dim=rec_dim,
                                            units=dynamics_units,
                                            layers=rec_layers)
                                )
    rec_dynamics_params = rec_dynamics.init(rng, *initialization_data_["rec_dynamics"])
    rec_dynamics_wrap = lambda x, t, params: rec_dynamics.apply(params, x, t)

    rec_to_gen = hk.transform(wrap_module(lambda: hk.Sequential([
        lambda x, y: jnp.concatenate((x, y), axis=-1),
        hk.Linear(100, **init_kwargs),
        jnp.tanh,
        hk.Linear(2 * gen_dim, **init_kwargs)
    ])))
    rec_to_gen_params = rec_to_gen.init(rng, *initialization_data_["rec_to_gen"])

    gen_dynamics = hk.transform(wrap_module(Dynamics,
                                            latent_dim=gen_dim,
                                            units=dynamics_units,
                                            layers=gen_layers))
    gen_dynamics_params = gen_dynamics.init(rng, *initialization_data_["gen_dynamics"])
    gen_dynamics_wrap = lambda x, t, params: gen_dynamics.apply(params, x, t)

    gen_to_data = hk.transform(wrap_module(hk.Linear,
                                           output_size=data_dim,
                                           **init_kwargs))
    gen_to_data_params = gen_to_data.init(rng, initialization_data_["gen_to_data"])

    init_params = {
        "gru": gru_params,
        "rec_dynamics": rec_dynamics_params,
        "rec_to_gen": rec_to_gen_params,
        "gen_dynamics": gen_dynamics_params,
        "gen_to_data": gen_to_data_params
    }

    def forward(params, data, timesteps, num_samples=3):
        """
        Forward pass of the model.
        y are the latent variables of the recognition model
        z are the latent variables of the generative model
        """
        # ode-rnn encoder
        final_y, final_y_std = jax.vmap(ode_rnn, in_axes=(None, 0, 0))(params, data, timesteps)

        # translate
        z0 = rec_to_gen.apply(params["rec_to_gen"], final_y, final_y_std)
        mean_z0, std_z0 = z0[..., :gen_dim], z0[..., gen_dim:]
        std_z0 = jnp.abs(std_z0)

        def sample_z0(key):
            """
            Sample generative latent variable using reparameterization trick.
            """
            return mean_z0 + std_z0 * jax.random.normal(key, shape=mean_z0.shape)
        z0 = jax.vmap(sample_z0)(jax.random.split(rng, num=num_samples))

        def integrate_sample(z0_):
            """
            Integrate one sample of z0 (for one batch).
            """
            z_, nfe_ = jax.vmap(lambda z_, t_: odeint(gen_dynamics_wrap, z_, t_,
                                                      params["gen_dynamics"], **gen_ode_kwargs))(z0_, timesteps)
            return z_[:, -1], nfe_
        z, nfe = jax.vmap(integrate_sample)(z0)

        # decode latent to data
        pred = jax.vmap(gen_to_data.apply, in_axes=(None, 0))(params["gen_to_data"], z)

        return pred

    def ode_rnn(params, data, timesteps):
        """
        ODE-RNN model.
        """

        init_t = timesteps[-1]
        init_y, init_std = gru.apply(params["gru"], jnp.zeros(rec_dim), jnp.zeros(rec_dim), data[-1])

        def scan_fun(carry, target):
            """
            Function to scan over observations of input sequence.
            """
            prev_y, prev_std, prev_t = carry
            xi, ti = target

            rev_dyn = lambda x, t, params_: -rec_dynamics_wrap(x, -t, params_)
            ys_ode, nfe = odeint(rev_dyn,
                                 prev_y,
                                 -jnp.array([prev_t, ti]),
                                 params["rec_dynamics"],
                                 **rec_ode_kwargs)
            yi_ode = ys_ode[-1]

            yi, yi_std = gru.apply(params["gru"],
                                   yi_ode, prev_std, xi)

            return (yi, yi_std, ti), (yi, yi_std)

        (final_y, final_y_std, _), latent_ys = lax.scan(scan_fun,
                                                        (init_y, init_std, init_t),
                                                        (data[-2::-1], timesteps[-2::-1]))

        return final_y, final_y_std

    model = {
        "forward": forward,
        "params": init_params
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


def loss_fn(forward, params, inputs, input_ts, targets):
    """
    The loss function for training.
    """
    preds, regs = forward(params, inputs, input_ts)
    loss_ = _loss_fn(preds, targets)
    reg_ = _reg_loss_fn(regs)
    weight_ = _weight_fn(params)
    return loss_ + lam * reg_ + lam_w * weight_


def init_toy_data():
    """
    Initialize toy data. This example is easier since time_points are shared across all examples.
    """
    n_samples = 1000
    noise_weight = 0.1

    timesteps, samples = Periodic1D(init_freq=None,
                                    init_amplitude=1.,
                                    final_amplitude=1.,
                                    final_freq=None,
                                    z0=1.).sample(rng,
                                                  n_samples=n_samples,
                                                  noise_weight=noise_weight)

    def _split_train_test(data, train_frac=0.8):
        data_train = data[:int(n_samples * train_frac)]
        data_test = data[int(n_samples * train_frac):]
        return data_train, data_test

    train_y, test_y = _split_train_test(samples)

    num_train = len(train_y)
    assert num_train % parse_args.batch_size == 0
    num_train_batches = num_train // parse_args.batch_size

    assert num_train % parse_args.test_batch_size == 0
    num_test_batches = num_train // parse_args.test_batch_size

    # make sure we always save the model on the last iteration
    assert num_train_batches * parse_args.nepochs % parse_args.save_freq == 0

    def gen_data(batch_size, shuffle=True, subsample=None):
        """
        Generator for train data.
        """
        key = rng
        num_batches = num_train // batch_size
        inds = jnp.arange(num_train)

        def swor(subkey, w, k):
            """
            Sample k items from collection of n items with weights given by w.
            """
            n = len(w)
            g = jax.random.gumbel(subkey, shape=(n,))
            g += jnp.log(w)
            g *= -1
            return jnp.argsort(g)[:k]

        def get_subsample(subkey, sample):
            """
            Subsample timeseries.
            """
            subsample_inds = jnp.sort(swor(subkey, jnp.ones_like(timesteps), subsample))
            return sample[subsample_inds], timesteps[subsample_inds]

        while True:
            if shuffle:
                key, = jax.random.split(key, num=1)
                epoch_inds = jax.random.shuffle(key, inds)
            else:
                epoch_inds = inds
            for i in range(num_batches):
                batch_inds = epoch_inds[i * batch_size: (i + 1) * batch_size]
                if subsample is not None:
                    # TODO: if we want to do proportional subsampling I don't think we can vmap
                    yield jax.vmap(get_subsample)(jax.random.split(key, num=batch_size), train_y[batch_inds])
                else:
                    yield train_y[batch_inds], jnp.repeat(timesteps[None], batch_size, axis=0)

    ds_train = gen_data(parse_args.batch_size, subsample=parse_args.subsample)
    ds_test = gen_data(parse_args.test_batch_size, shuffle=False)

    meta = {
        "num_batches": num_train_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test, meta


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    ds_train, ds_test, meta = init_toy_data()
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    model = init_model({}, {})
    forward = model["forward"]
    batch = next(ds_train)
    result = forward(model["params"], *batch)

    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    def lr_schedule(train_itr):
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 60, 1e-1, id, 0,
                        lambda _: lax.cond(_epoch < 100, 1e-2, id, 0,
                                           lambda _: lax.cond(_epoch < 140, 1e-3, id, 1e-4, id)))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)
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
        inputs, input_ts, targets = _batch
        preds, regs = forward(params, inputs, input_ts)
        loss_ = _loss_fn(preds, targets)
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

    itr = 0
    info = collections.defaultdict(dict)
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)

            itr += 1

            opt_state = update(itr, opt_state, batch)

            if itr % parse_args.test_freq == 0:
                loss_aug_, loss_, loss_reg_, nfe_ = evaluate_loss(opt_state, ds_train_eval)

                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f} | NFE {:.6f}'.format(itr, loss_aug_, loss_, loss_reg_, nfe_)

                print(print_str)

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
