"""
FFJORD on MNIST, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import sys

import datasets

import haiku as hk

import jax
from jax import lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import odeint, odeint_sepaux
from jax.experimental.jet import jet

from jax.config import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=500)  # TODO: just a guess
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=1e-6)
parser.add_argument('--atol', type=float, default=1.4e-8)  # 1e-8 (original values)
parser.add_argument('--rtol', type=float, default=1.4e-8)  # 1e-6
parser.add_argument('--method', type=str, default="dopri5")
parser.add_argument('--no_vmap', action="store_true")
parser.add_argument('--init_step', type=float, default=1.)
parser.add_argument('--reg', type=str, choices=['none', 'r2', 'r3', 'r4'], default='none')
parser.add_argument('--test_freq', type=int, default=300)
parser.add_argument('--save_freq', type=int, default=300)
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_count_nfe', action="store_true")
parser.add_argument('--ckpt_freq', type=int, default=300)
parser.add_argument('--ckpt_path', type=str, default="./ck.pt")
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hdim_factor', type=int, default=20)
parser.add_argument('--nonlinearity', type=str, default="softplus")
parse_args = parser.parse_args()


assert os.path.exists(parse_args.dirname)

# set up config

reg = parse_args.reg
lam = parse_args.lam
lam_w = parse_args.lam_w
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
dirname = parse_args.dirname
count_nfe = not parse_args.no_count_nfe
vmap = False if parse_args.no_vmap is True else True
vmap = False
num_blocks = 0
ode_kwargs = {
    "atol": parse_args.atol,
    "rtol": parse_args.rtol,
    # "method": parse_args.method,
    # "init_step": parse_args.init_step
}

# jax.nn.softplus jet will fail because of convert element type
# logaddexp will fail because no lax.max primitive implemented
# TODO: maybe use jax.nn.softplus when not doing jet?
# softplus = jax.nn.softplus
softplus = lambda x: jnp.where(x >= 0,
                               x + jnp.log1p(jnp.exp(-x)),
                               jnp.log1p(jnp.exp(x)))

def sigmoid(z):
  """
  Numerically stable sigmoid.
  """
  return 1/(1 + jnp.exp(-z))


nonlinearity = softplus if parse_args.nonlinearity == "softplus" else jnp.tanh


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  # TODO: numerically zero? wtf??
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

  return (jnp.reshape(y0[:-1], z_shape), [jnp.reshape(y1[:-1], z_shape)])


# set up modules
class ConcatSquashLinear(hk.Module):
    """
    ConcatSquash Linear layer.
    """
    def __init__(self, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = hk.Linear(dim_out)
        self._hyper_bias = hk.Linear(dim_out, with_bias=False)
        self._hyper_gate = hk.Linear(dim_out)

    def __call__(self, x, t):
        thing1 = self._layer(x)
        thing = self._layer(x) * sigmoid(self._hyper_gate(jnp.reshape(t, (1, 1)))) \
               + self._hyper_bias(jnp.reshape(t, (1, 1)))
        return self._layer(x) * sigmoid(self._hyper_gate(jnp.reshape(t, (1, 1)))) \
               + self._hyper_bias(jnp.reshape(t, (1, 1)))


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # normal
    return jax.random.normal(key, shape)
    # rademacher
    # return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float64) * 2 - 1


class NN_Dynamics(hk.Module):
    """
    NN_Dynamics of the ODENet.
    """

    def __init__(self,
                 hidden_dims,
                 input_shape):
        super(NN_Dynamics, self).__init__()
        self.input_shape = input_shape
        layers = []
        activation_fns = []
        base_layer = ConcatSquashLinear

        for dim_out in hidden_dims + (input_shape[-1], ):
            layer = base_layer(dim_out)
            layers.append(layer)
            activation_fns.append(nonlinearity)

        self.layers = layers
        self.activation_fns = activation_fns[:-1]

    def __call__(self, x, t):
        x = jnp.reshape(x, (-1, *self.input_shape))
        dx = x
        for l, layer in enumerate(self.layers):
            dx = layer(dx, t)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


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


def initialization_data(input_shape):
    """
    Data for initializing the modules.
    """
    # use the batch size to allocate memory
    input_shape = (parse_args.test_batch_size, ) + input_shape[1:]
    data = {
        "ode": aug_init(jnp.zeros(input_shape))[:-2] + (0., )
    }
    return data


def init_model(n_dims):
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (-1, n_dims)

    initialization_data_ = initialization_data(input_shape)

    dynamics = hk.transform(wrap_module(NN_Dynamics,
                                        input_shape=input_shape[1:],
                                        hidden_dims=(n_dims * parse_args.hdim_factor, ) * parse_args.num_layers))
    dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
    dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)

    def reg_dynamics(y, t, params):
        """
        NN_Dynamics of regularization for ODE integration.
        """
        if reg == "none":
            y = jnp.reshape(y, input_shape)
            return jnp.zeros((y.shape[0], 1))
        else:
            # TODO: keep second axis at 1?
            # do r3 regularization
            y0, y_n = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
            r = y_n[-1]
            return jnp.mean(jnp.square(r), axis=[axis_ for axis_ in range(1, r.ndim)])

    def ffjord_dynamics(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state.
        """
        y, p = yp
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return dy, -div

    def aug_dynamics(ypr, t, eps, params):
        """
        NN_Dynamics augmented with logp and regularization.
        """
        y, p, r = ypr

        dy, dp = ffjord_dynamics((y, p), t, eps, params)
        dr = reg_dynamics(y, t, params)
        return dy, dp, dr
    nodeint_aux = lambda y0, ts, eps, params: odeint_sepaux(lambda y, t, eps, params: dynamics_wrap(y, t, params),
                                                            aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]
    # nodeint_aux = lambda y0, ts, eps, params: odeint(aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]
    nodeint = lambda y0, ts, eps, params: odeint(aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]

    def ode_aux(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, rs = nodeint_aux(reg_init(y, delta_logp), ts, eps, params)
        return ys[-1], delta_logps[-1], rs[-1]

    def ode(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, rs = nodeint(reg_init(y, delta_logp), ts, eps, params)
        return ys[-1], delta_logps[-1], rs[-1]

    if count_nfe:
        # TODO: w/ finlay trick this is not true NFE
        if vmap:
            unreg_nodeint = jax.vmap(lambda z, delta_logp, t, eps, params:
                                     odeint(ffjord_dynamics, (z, delta_logp), t, eps, params, **ode_kwargs)[1],
                                     (0, 0, None, 0, None))
        else:
            unreg_nodeint = lambda z, delta_logp, t, eps, params: \
                odeint(ffjord_dynamics, (z, delta_logp), t, eps, params, **ode_kwargs)[1]

        @jax.jit
        def nfe_fn(key, params, _x):
            """
            Function to return NFE.
            """
            eps = get_epsilon(key, _x.shape)

            f_nfe = unreg_nodeint(*aug_init(_x)[:-1], ts, eps, params["ode"])
            return jnp.mean(f_nfe)

    else:
        nfe_fn = None

    def forward_aux(key, params, _x):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _x.shape)

        z, delta_logp, regs = ode_aux(params["ode"], *aug_init(_x)[:-1], eps)

        return z, delta_logp, regs

    def forward(key, params, _x):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _x.shape)

        z, delta_logp, regs = ode_aux(params["ode"], *aug_init(_x)[:-1], eps)

        return z, delta_logp, regs

    model = {"model": {
        "ode": ode
    }, "params": {
        "ode": dynamics_params
    }, "nfe": nfe_fn,
        "forward": forward
    }

    return forward_aux, model


def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    """
    batch_size = y.shape[0]
    return y, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))


def reg_init(y, delta_logp):
    """
    Initialize dynamics with 0 for and regs.
    """
    batch_size = y.shape[0]
    return y, delta_logp, jnp.zeros((batch_size, 1))


def _acc_fn(logits, labels):
    """
    Classification accuracy of the model.
    """
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)


def standard_normal_logprob(z):
    """
    Log probability of standard normal.
    """
    logz = -0.5 * jnp.log(2 * jnp.pi)
    return logz - jnp.square(z) / 2


def _loss_fn(z, delta_logp):
    logpz = jnp.sum(jnp.reshape(standard_normal_logprob(z), (z.shape[0], -1)), axis=1, keepdims=True)  # logp(z)
    logpx = logpz - delta_logp

    return -jnp.mean(logpx)  # likelihood in nats


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, images, key):
    """
    The loss function for training.
    """
    z, delta_logp, regs = forward(key, params, images)
    loss_ = _loss_fn(z, delta_logp)
    reg_ = _reg_loss_fn(regs)
    weight_ = _weight_fn(params)
    return loss_ + lam * reg_ + lam_w * weight_


def init_data():
    """
    Initialize data.
    """
    data = datasets.MINIBOONE()

    num_train = data.trn.N
    num_test = data.trn.N
    # num_test = data.tst.N

    data.trn.x = jnp.float64(data.trn.x)
    data.tst.x = jnp.float64(data.tst.x)

    num_batches = num_train // parse_args.batch_size + 1 * (num_train % parse_args.batch_size != 0)
    num_test_batches = num_test // parse_args.test_batch_size + 1 * (num_train % parse_args.test_batch_size != 0)

    # make sure we always save the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0

    def gen_train_data():
        """
        Generator for train data.
        """
        key = rng
        inds = jnp.arange(num_train)

        while True:
            key, = jax.random.split(key, num=1)
            epoch_inds = jax.random.shuffle(key, inds)
            for i in range(num_batches):
                batch_inds = epoch_inds[i * parse_args.batch_size: min((i + 1) * parse_args.batch_size, num_train)]
                yield data.trn.x[batch_inds]

    def gen_test_data():
        """
        Generator for train data.
        """
        inds = jnp.arange(num_test)
        while True:
            for i in range(num_test_batches):
                batch_inds = inds[i * parse_args.test_batch_size: min((i + 1) * parse_args.test_batch_size, num_test)]
                yield data.trn.x[batch_inds]

    ds_train = gen_train_data()
    ds_test = gen_test_data()

    meta = {
        "dims": data.n_dims,
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test, meta


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    # init the model first so that jax gets enough GPU memory before TFDS
    forward, model = init_model(43)  # how do you sleep at night
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    ds_train, ds_test_eval, meta = init_data()
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    opt_init, opt_update, get_params = optimizers.adam(step_size=parse_args.lr)
    unravel_opt = ravel_pytree(opt_init(model["params"]))[1]
    if os.path.exists(parse_args.ckpt_path):
        outfile = open(parse_args.ckpt_path, 'rb')
        state_dict = pickle.load(outfile)
        outfile.close()

        opt_state = unravel_opt(state_dict["opt_state"])

        load_itr = state_dict["itr"]
    else:
        init_params = model["params"]
        opt_state = opt_init(init_params)

        load_itr = 0

    @jax.jit
    def update(_itr, _opt_state, _key, _batch):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(_itr, grad_fn(get_params(_opt_state), _batch, _key), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, _key):
        """
        Convenience function for calculating losses separately.
        """
        z, delta_logp, regs = model["forward"](_key, get_params(_opt_state), _batch)
        loss_ = _loss_fn(z, delta_logp)
        reg_ = _reg_loss_fn(regs)
        total_loss_ = loss_ + lam * reg_
        return total_loss_, loss_, reg_

    def evaluate_loss(opt_state, _key, ds_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_loss_aug_, sep_loss_, sep_loss_reg_, nfe, bs = [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            _key, = jax.random.split(_key, num=1)
            test_batch = next(ds_eval)

            test_batch_loss_aug_, test_batch_loss_, test_batch_loss_reg_ = sep_losses(opt_state, test_batch, _key)

            if count_nfe:
                nfe.append(model["nfe"](_key, get_params(opt_state), test_batch))
            else:
                nfe.append(0)

            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_reg_.append(test_batch_loss_reg_)
            bs.append(len(test_batch))

        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_reg_ = jnp.array(sep_loss_reg_)
        nfe = jnp.array(nfe)
        bs = jnp.array(bs)

        return jnp.average(sep_loss_aug_, weights=bs), \
               jnp.average(sep_loss_, weights=bs), \
               jnp.average(sep_loss_reg_, weights=bs), \
               jnp.average(nfe, weights=bs)

    itr = 0
    info = collections.defaultdict(dict)

    key = rng

    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            key, = jax.random.split(key, num=1)
            batch = next(ds_train)

            itr += 1

            if itr <= load_itr:
                continue

            opt_state = update(itr, opt_state, key, batch)

            if itr % parse_args.test_freq == 0:
                loss_aug_, loss_, loss_reg_, nfe_ = evaluate_loss(opt_state, key, ds_test_eval)

                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f} | NFE {:.6f}'.format(itr, loss_aug_, loss_, loss_reg_, nfe_)

                print(print_str)

                outfile = open("%s/reg_%s_lam_%.18e_num_blocks_%d_info.txt" % (dirname, reg, lam, num_blocks), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["loss_aug"] = loss_aug_
                info[itr]["loss"] = loss_
                info[itr]["loss_reg"] = loss_reg_
                info[itr]["nfe"] = nfe_

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_lam_%.18e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()

            outfile = open("%s/reg_%s_lam_%.18e_num_blocks_%d_iter.txt" % (dirname, reg, lam, num_blocks), "a")
            outfile.write("Iter: {:04d}\n".format(itr))
            outfile.close()

            if itr % parse_args.ckpt_freq == 0:
                state_dict = {
                    "opt_state": ravel_pytree(opt_state)[0],
                    "itr": itr,
                }
                outfile = open(parse_args.ckpt_path, 'wb')
                pickle.dump(state_dict, outfile)
                outfile.close()
    meta = {
        "info": info,
        "args": parse_args
    }
    outfile = open("%s/reg_%s_lam_%.18e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "wb")
    pickle.dump(meta, outfile)
    outfile.close()


if __name__ == "__main__":
    run()
