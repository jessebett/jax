"""
Profiling ODENets to see bottlenecks in performance. Reimplemented without using weird parameter searching.
"""
import argparse
import os
import pickle
import sys
import time

import haiku as hk
import numpy as onp
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.ode import build_odeint, odeint, vjp_odeint

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=2)
parser.add_argument('--nepochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none', 'r3'], default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resnet', action="store_true")
parser.add_argument('--count_nfe', action="store_true")
parser.add_argument('--num_blocks', type=int, default=6)
parse_args = parser.parse_args()


assert os.path.exists(parse_args.dirname)

# set up config

reg = parse_args.reg
lam = parse_args.lam
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
dirname = parse_args.dirname
odenet = False if parse_args.resnet is True else True
count_nfe = True if parse_args.count_nfe is True else False
num_blocks = parse_args.num_blocks


# some primitive functions
def sigmoid(z):
  """
  Defined using only numpy primitives (but probably less numerically stable).
  """
  return 1./(1. + jnp.exp(-z))


def softmax_cross_entropy(logits, labels):
  """
  Cross-entropy loss applied to softmax.
  """
  one_hot = hk.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  def g(z):
    """
    Closure to expand z.
    """
    return f(z, t)

  (y0, [y1h]) = jax.jet(g, (z, ), ((jnp.ones_like(z), ), ))
  (y0, [y1, y2h]) = jax.jet(g, (z, ), ((y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jax.jet(g, (z, ), ((y0, y1, y2h), ))

  return (y0, [y1, y2])


# set up modules
class Flatten(hk.Module):
    """
    Flatten all dimensions except batch dimension.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, x):
        return jnp.reshape(x, (x.shape[0], -1))


class ConcatConv2d(hk.Module):
    """
    Convolution with extra channel and skip connection for time
    .
    """

    def __init__(self, **kwargs):
        super(ConcatConv2d, self).__init__()
        self._layer = hk.Conv2D(**kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


def aug_init(y):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    return jnp.concatenate((jnp.ravel(y), jnp.zeros(y.shape[0])))


def unpack_aug(yr, ode_dim):
    """
    Unpack the dynamics according to the structure in aug_init.
    """
    batch_size = yr.size // (ode_dim + 1)
    return yr[:-batch_size], yr[-batch_size:]


class ResBlock(hk.Module):
    """
    Standard ResBlock.
    """

    def __init__(self, output_channels):
        super(ResBlock, self).__init__()
        self.conv = hk.Conv2D(output_channels=output_channels,
                              kernel_shape=3,
                              stride=1,
                              padding="SAME")

    def __call__(self, x):
        f_x = sigmoid(x)
        f_x = self.conv(f_x)
        return x + f_x


def _acc_fn(logits, labels):
    """
    Classification accuracy of the model.
    """
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)


def _loss_fn(logits, labels):
    return jnp.mean(softmax_cross_entropy(logits, labels))


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def wrap_module(module, *module_args):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args):
        """
        Wrapping of module.
        """
        model = module(*module_args)
        return model(*args)
    return wrap


class PreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self):
        super(PreODE, self).__init__()
        self.model = hk.Sequential([
            lambda x: x.astype(jnp.float32) / 255.,
            hk.Conv2D(output_channels=8,
                      kernel_shape=3,
                      stride=2,
                      padding=lambda x: (1, 1)),
            hk.AvgPool(window_shape=(1, 11, 11, 1),
                       strides=(1, 1, 1, 1),
                       padding="VALID")
        ])

    def __call__(self, x):
        return self.model(x)


class Dynamics(hk.Module):
    """
    Dynamics of the ODENet.
    """

    def __init__(self, input_shape):
        super(Dynamics, self).__init__()
        self.input_shape = input_shape
        output_channels = input_shape[-1]
        self.concat_conv = ConcatConv2d(output_channels=output_channels,
                                        kernel_shape=3,
                                        stride=1,
                                        padding="SAME")

    def __call__(self, x, t):
        x = jnp.reshape(x, self.input_shape)
        x = sigmoid(x)
        x = self.concat_conv(x, t)
        x = jnp.ravel(x)
        return x


class PostODE(hk.Module):
    """
    Module applied after the ODE layer.
    """

    def __init__(self):
        super(PostODE, self).__init__()
        self.avg_pool = hk.AvgPool(window_shape=(1, 3, 3, 1),
                                   strides=(1, 1, 1, 1),
                                   padding="VALID")
        self.flatten = Flatten()
        self.linear = hk.Linear(10)

    def __call__(self, x):
        x = sigmoid(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


def initialization_data(input_shape, ode_shape):
    """
    Data for initializing the modules.
    """
    ode_shape = (1, ) + ode_shape[1:]
    data = {
        "pre_ode": jnp.zeros(input_shape),
        "ode": (jnp.zeros(ode_shape), 0.),
        "post_ode": jnp.zeros(ode_shape)
    }
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (1, 28, 28, 1)
    ode_shape = (-1, 4, 4, 8)
    ode_dim = onp.prod(ode_shape[1:])

    initialization_data_ = initialization_data(input_shape, ode_shape)

    pre_ode = hk.transform(wrap_module(PreODE))
    pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])

    dynamics = hk.transform(wrap_module(Dynamics, ode_shape))
    dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)
    dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
    if reg:
        def reg_dynamics(y, t, params):
            """
            Dynamics of regularization for ODE integration.
            """
            batch_size = y.size // ode_dim
            if reg == "none":
                return jnp.zeros(batch_size)
            else:
                # do r3 regularization
                y0, y_n = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
                r = jnp.reshape(y_n[-1], (-1, ode_dim))
                return jnp.sum(r ** 2, axis=1)

        def aug_dynamics(yr, t, params):
            """
            Dynamics augmented with regularization.
            """
            y, r = unpack_aug(yr, ode_dim)
            dydt = dynamics_wrap(y, t, params)
            drdt = reg_dynamics(y, t, params)
            return jnp.concatenate((dydt, drdt))
        nodeint = build_odeint(aug_dynamics)
    else:
        nodeint = build_odeint(dynamics_wrap)

    def ode(params, out_pre_ode):
        """
        Apply the ODE block.
        """
        flat_out_ode = nodeint(aug_init(out_pre_ode), ts, params)[-1]
        out_ode, out_ode_r = unpack_aug(flat_out_ode, ode_dim)
        out_ode = jnp.reshape(out_ode, ode_shape)
        return out_ode, out_ode_r

    if count_nfe:
        unreg_nodeint = lambda y0, t, params: odeint(dynamics_wrap, y0, t, params)
        # 2 corresponds to number of integration bounds
        unreg_nodeint_vjp = lambda cotangent, y0, t, params: \
            vjp_odeint(dynamics_wrap, y0, t, params, nfe=True)[1](jnp.reshape(cotangent, (2, cotangent.size // 2)))[-1]

        @jax.jit
        def nfe_fn(params, _images, _labels):
            """
            Function to return NFE.
            """
            out_1 = pre_ode.apply(params["pre_ode"], _images)
            in_ode = jnp.ravel(out_1)
            flat_out_ode, f_nfe = unreg_nodeint(in_ode, ts, params["ode"])
            out_ode = jnp.reshape(flat_out_ode[-1], ode_shape)

            def partial_loss(_out_ode, targets):
                """
                Evaluates loss wrt output of ODEBlock. (for b-nfe calculations).
                """
                preds = post_ode.apply(params["post_ode"], _out_ode)
                return _loss_fn(preds, targets)

            grad_partial_loss_ = jax.grad(partial_loss)(out_ode, _labels)
            # grad is 0 at t0 (since always equal)
            cotangent = jnp.stack((jnp.zeros_like(grad_partial_loss_), grad_partial_loss_), axis=0)
            b_nfe = unreg_nodeint_vjp(cotangent, jnp.reshape(in_ode, (-1, )), ts, params["ode"])

            return f_nfe, b_nfe

    else:
        nfe_fn = None

    post_ode = hk.transform(wrap_module(PostODE))
    post_ode_params = post_ode.init(rng, initialization_data_["post_ode"])

    # return a dictionary of the three components of the model
    model = {
        "model": {
            "pre_ode": pre_ode.apply,
            "ode": ode,
            "post_ode": post_ode.apply
        },
        "params": {
            "pre_ode": pre_ode_params,
            "ode": dynamics_params,
            "post_ode": post_ode_params
        },
        "nfe": nfe_fn
    }

    def forward(params, _images):
        """
        Forward pass of the model.
        """
        model_ = model["model"]
        out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
        out_ode, regs = model_["ode"](params["ode"], out_pre_ode)
        logits = model_["post_ode"](params["post_ode"], out_ode)

        return logits, regs

    return forward, model


def loss_fn(forward, params, images, labels):
    """
    The loss function for training.
    """
    logits, regs = forward(params, images)
    loss_ = _loss_fn(logits, labels)
    reg_ = _reg_loss_fn(regs)
    return loss_ + lam * reg_


def init_data():
    """
    Initialize data.
    """
    (ds_train,), ds_info = tfds.load('mnist',
                                     split=['train'],
                                     shuffle_files=True,
                                     as_supervised=True,
                                     with_info=True)

    num_train = ds_info.splits['train'].num_examples

    assert num_train % parse_args.batch_size == 0
    num_batches = num_train // parse_args.batch_size

    # make sure we always save the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0

    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(1000)
    ds_train, ds_train_eval = ds_train.batch(parse_args.batch_size), ds_train.batch(parse_args.test_batch_size)
    ds_train, ds_train_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_train_eval)

    meta = {
        "num_train": num_train,
        "num_batches": num_batches
    }

    return ds_train, ds_train_eval, meta


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    ds_train, ds_train_eval, meta = init_data()
    num_train = meta["num_train"]
    num_batches = meta["num_batches"]

    forward, model = init_model()

    opt_init, opt_update, get_params = optimizers.adam(step_size=parse_args.lr)
    opt_state = opt_init(model["params"])

    @jax.jit
    def update(_itr, _opt_state, _batch):
        """
        Update the params based on grad for current batch.
        """
        images, labels = _batch
        grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))
        return opt_update(_itr, grad_fn(get_params(_opt_state), images, labels), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        images, labels = _batch
        logits, regs = forward(params, images)
        loss_ = _loss_fn(logits, labels)
        reg_ = _reg_loss_fn(regs)
        total_loss_ = loss_ + lam * reg_
        acc_ = _acc_fn(logits, labels)
        return acc_, total_loss_, loss_, reg_

    def evaluate_loss(opt_state, ds_train_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        test_batch_size = parse_args.test_batch_size if odenet else num_train
        num_test_batches = num_train // test_batch_size
        sep_acc_, sep_loss_aug_, sep_loss_, sep_loss_reg_ = [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)

            test_batch_acc_, test_batch_loss_aug_, test_batch_loss_, test_batch_loss_reg_ = \
                sep_losses(opt_state, test_batch)

            sep_acc_.append(test_batch_acc_)
            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_reg_.append(test_batch_loss_reg_)

        sep_acc_ = jnp.array(sep_acc_)
        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_reg_ = jnp.array(sep_loss_reg_)

        return jnp.mean(sep_acc_), jnp.mean(sep_loss_aug_), jnp.mean(sep_loss_), jnp.mean(sep_loss_reg_)

    itr = 0
    iter_times = []
    batch_times = []
    update_times = []
    test_times = []
    save_times = []
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            iter_start = time.time()
            batch_start = time.time()
            batch = next(ds_train)
            batch_end = time.time()

            itr += 1

            update_start = time.time()
            opt_state = update(itr, opt_state, batch)
            update_end = time.time()

            test_start = time.time()
            if itr % parse_args.test_freq == 0:
                acc_, loss_aug_, loss_, loss_reg_ = evaluate_loss(opt_state, ds_train_eval)

                print_str = 'Iter {:04d} | Accuracy {:.6f} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f}'.format(itr, acc_, loss_aug_, loss_, loss_reg_)

                print(print_str)
                print(print_str, file=sys.stderr)

                if count_nfe:
                    f_nfe, b_nfe = model["nfe"](get_params(opt_state), *batch)
                    print(f_nfe, b_nfe)
            test_end = time.time()

            save_start = time.time()
            if itr % parse_args.save_freq == 0:
                if odenet:
                    param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                else:
                    param_filename = "%s/reg_%s_lam_%.4e_num_blocks_%d_%d_fargs.pickle" % (dirname, reg, lam, num_blocks, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()
            save_end = time.time()
            iter_end = time.time()

            iter_times.append(iter_end - iter_start)
            batch_times.append(batch_end - batch_start)
            update_times.append(update_end - update_start)
            test_times.append(test_end - test_start)
            save_times.append(save_end - save_start)

    times = {
        "iter": iter_times,
        "batch": batch_times,
        "update": update_times,
        "test": test_times,
        "save": save_times
    }
    outfile = open("%s/times.pickle" % dirname, "wb")
    pickle.dump(times, outfile)
    outfile.close()


if __name__ == "__main__":
    run()
