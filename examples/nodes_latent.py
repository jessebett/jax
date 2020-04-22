"""
Implementing Latent ODE!! YEET
"""
import argparse
import collections
import os
import pickle
import sys
from functools import partial

import haiku as hk

from timeseries import init_periodic_data

import jax
from jax import lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=500)
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
parser.add_argument('--test_freq', type=int, default=1)
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
lam_rec = lam_gen = lam
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
    ODE dynamics for both encoder and decoder.
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


def augment_dynamics(dynamics, sign=1):
    """
    Closure to augment dynamics.
    """
    # TODO: check if stuff is cached here
    def reg_dynamics(y, t, params):
        """
        Dynamics of regularization.
        """
        if reg == "none":
            return 0.
        else:
            # do r3 regularization
            y0, y_n = sol_recursive(lambda _y, _t: dynamics(_y, _t, params), y, t)
            r = y_n[-1]
            return jnp.sum(r ** 2, axis=[axis_ for axis_ in range(1, r.ndim)])

    def aug_dynamics(yr, t, params):
        """
        Dynamics augmented with regularization.
        """
        y, r = yr
        dydt = dynamics(y, t, params)
        drdt = reg_dynamics(y, t, params)
        return sign * dydt, sign * drdt
    return aug_dynamics


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

    def forward(count_nfe_, params, data, timesteps, num_samples=3):
        """
        Forward pass of the model.
        y are the latent variables of the recognition model
        z are the latent variables of the generative model
        """
        # ode-rnn encoder
        final_y, final_y_std, rec_r, rec_nfes = \
            jax.vmap(partial(ode_rnn, count_nfe_), in_axes=(None, 0, 0))(params, data, timesteps)

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
            if count_nfe_:
                dynamics = gen_dynamics_wrap
                init_fn = lambda x: x
            else:
                dynamics = augment_dynamics(gen_dynamics_wrap)
                init_fn = aug_init
            return jax.vmap(lambda z_, t_: odeint(dynamics, init_fn(z_), t_,
                                                  params["gen_dynamics"], **gen_ode_kwargs))(z0_, timesteps)
        z_r, gen_nfes = jax.vmap(integrate_sample)(z0)
        if count_nfe_:
            z, gen_r = z_r, 0.
        else:
            z, gen_r = z_r

        # decode latent to data, vmapping over batch and timepoints
        pred = jax.vmap(jax.vmap(partial(gen_to_data.apply, params["gen_to_data"]), in_axes=1, out_axes=1))(z)

        z0_params = {
            "mean": mean_z0,
            "std": std_z0
        }

        nfe = {
            "rec": jnp.mean(jnp.sum(rec_nfes, axis=1)),
            "gen": jnp.mean(gen_nfes)
        }

        return pred, rec_r, gen_r, z0_params, nfe

    def ode_rnn(count_nfe_, params, data, timesteps):
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

            if count_nfe_:
                dynamics = lambda *args: -rec_dynamics_wrap(*args)
                init = prev_y
            else:
                dynamics = augment_dynamics(rec_dynamics_wrap, sign=-1)
                init = aug_init(prev_y)
            ys_ode, nfe = odeint(lambda x, t, params_: dynamics(x, -t, params_),
                                 init,
                                 -jnp.array([prev_t, ti]),
                                 params["rec_dynamics"],
                                 **rec_ode_kwargs)
            if count_nfe_:
                yi_ode = ys_ode[-1]
                r_ode = 0.
            else:
                ys_ode, rs_ode = ys_ode
                yi_ode, r_ode = ys_ode[-1], rs_ode[-1]

            yi, yi_std = gru.apply(params["gru"],
                                   yi_ode, prev_std, xi)

            return (yi, yi_std, ti), (r_ode, nfe)

        (final_y, final_y_std, _), rs_nfes = lax.scan(scan_fun,
                                                      (init_y, init_std, init_t),
                                                      (data[-2::-1], timesteps[-2::-1]))

        rs, nfes = rs_nfes

        # TODO: if we regularize encoder, how to weight rs in general?
        return final_y, final_y_std, jnp.mean(rs), nfes

    model = {
        "forward": partial(forward, False),
        "params": init_params,
        "nfe": lambda *args: partial(forward, count_nfe)(*args)[-1]
    }

    return model


def aug_init(y):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    return y, 0.


def _likelihood(preds, data):
    """
    Compute log-likelihood of data under current predictions.
    """
    def sample_likelihood(data_, mu, std=0.01):
        """
        Log-Likelihood of one sample.
        """
        return -((data_ - mu) ** 2) / (2 * std ** 2) - jnp.log(std) - jnp.log(2 * jnp.pi) / 2
    return jnp.mean(sample_likelihood(data[None], preds), axis=[1, 2, 3])


def _kl_div(params):
    """
    Analytically compute KL between z0 distribution and prior.
    """
    mean_prior = 0
    std_prior = 1
    var_ratio = (params["std"] / std_prior) ** 2
    t1 = ((params["mean"] - mean_prior) / std_prior) ** 2
    return jnp.mean(0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio)))


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, data, timesteps, kl_coef):
    """
    The loss function for training.
    """
    preds, rec_r, gen_r, z0_params, nfe = forward(params, data, timesteps)
    likelihood_ = _likelihood(preds, data)
    kl_ = _kl_div(z0_params)
    rec_reg_ = _reg_loss_fn(rec_r)
    gen_reg_ = _reg_loss_fn(gen_r)
    # weight_ = _weight_fn(params)
    return -jnp.mean(likelihood_ - kl_coef * kl_) + lam_rec * rec_reg_ + lam_gen * gen_reg_
    # return loss_ + lam * reg_ + lam_w * weight_


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    ds_train, ds_test, meta = init_periodic_data(rng, parse_args)
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    model = init_model({}, {})
    forward = model["forward"]
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    lr_schedule = optimizers.exponential_decay(step_size=parse_args.lr,
                                               decay_steps=1,
                                               decay_rate=0.999,
                                               lowest=parse_args.lr / 10)
    opt_init, opt_update, get_params = optimizers.adamax(step_size=lr_schedule)
    opt_state = opt_init(model["params"])

    def get_kl_coef(epoch_):
        """
        Tuning schedule for KL coefficient.
        """
        return max(0., 1 - 0.99 ** (epoch_ - 10))

    @jax.jit
    def update(_itr, _opt_state, _batch, kl_coef):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(_itr, grad_fn(get_params(_opt_state), *_batch, kl_coef), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, kl_coef):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        data, timesteps = _batch
        preds, rec_r, gen_r, z0_params, nfe = forward(params, data, timesteps)
        likelihood_ = _likelihood(preds, data)
        kl_ = _kl_div(z0_params)
        return -jnp.mean(likelihood_ - kl_coef * kl_), likelihood_, kl_, rec_r, gen_r

    def evaluate_loss(opt_state, ds_test, kl_coef):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        # TODO: refactor as scan
        loss, likelihood, kl, rec_r, gen_r, rec_nfe, gen_nfe = [], [], [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_test)

            batch_loss, batch_likelihood, batch_kl, batch_rec_r, batch_gen_r = \
                sep_losses(opt_state, test_batch, kl_coef)

            if count_nfe:
                nfes = model["nfe"](get_params(opt_state), *test_batch)
                rec_nfe.append(nfes["rec"])
                gen_nfe.append(nfes["gen"])
            else:
                rec_nfe.append(0)
                gen_nfe.append(0)

            loss.append(batch_loss)
            likelihood.append(batch_likelihood)
            kl.append(batch_kl)
            rec_r.append(batch_rec_r)
            gen_r.append(batch_gen_r)

        loss = jnp.array(loss)
        likelihood = jnp.array(likelihood)
        kl = jnp.array(kl)
        rec_r = jnp.array(rec_r)
        gen_r = jnp.array(gen_r)
        rec_nfe = jnp.array(rec_nfe)
        gen_nfe = jnp.array(gen_nfe)

        return jnp.mean(loss), jnp.mean(likelihood), jnp.mean(kl), jnp.mean(rec_r), jnp.mean(gen_r), \
               jnp.mean(rec_nfe), jnp.mean(gen_nfe)

    itr = 0
    info = collections.defaultdict(dict)
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)

            itr += 1

            opt_state = update(itr, opt_state, batch, get_kl_coef(epoch))

            if itr % parse_args.test_freq == 0:
                loss_, likelihood_, kl_, rec_r_, gen_r_, rec_nfe_, gen_nfe_ = \
                    evaluate_loss(opt_state, ds_test, get_kl_coef(epoch))

                print_str = 'Iter {:04d} | Loss {:.6f} | ' \
                            'Likelihood {:.6f} | KL {:.6f} | Enc. r {:.6f} | Dec. r {:.6f} | ' \
                            'Enc. NFE {:.6f} | Dec. NFE {:.6f}'.\
                    format(itr, loss_, likelihood_, kl_, rec_r_, gen_r_, rec_nfe_, gen_nfe_)

                print(print_str)

                outfile = open("%s/reg_%s_lam_%.4e_num_blocks_%d_info.txt" % (dirname, reg, lam, num_blocks), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["loss"] = loss_
                info[itr]["likelihood"] = likelihood_
                info[itr]["kl"] = kl_
                info[itr]["rec_r"] = rec_r_
                info[itr]["gen_r"] = gen_r_
                info[itr]["rec_nfe"] = rec_nfe_
                info[itr]["gen_nfe"] = gen_nfe_

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
