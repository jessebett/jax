"""
Neural ODEs on MNIST with no downsampling before ODE, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import sys

import haiku as hk
import tensorflow_datasets as tfds

import jax
from jax import lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1e-8)
parser.add_argument('--rtol', type=float, default=1e-8)
parser.add_argument('--method', type=str, default="dopri5")
parser.add_argument('--no_vmap', action="store_true")
parser.add_argument('--init_step', type=float, default=1.)
parser.add_argument('--reg', type=str, choices=['none', 'r3'], default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=3000)
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

  (y0, [y1h]) = jet(g, (z, ), ((jnp.ones_like(z), ), ))
  (y0, [y1, y2h]) = jet(g, (z, ), ((y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jet(g, (z, ), ((y0, y1, y2h), ))

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


class ConcatConv2D(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, **kwargs):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(**kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


class ResBlock(hk.Module):
    """
    Standard ResBlock.
    """

    def __init__(self, output_channels):
        super(ResBlock, self).__init__()
        self.conv1 = hk.Conv2D(output_channels=output_channels,
                               kernel_shape=3,
                               stride=1,
                               padding=lambda _: (1, 1),
                               with_bias=False)
        self.conv2 = hk.Conv2D(output_channels=output_channels,
                               kernel_shape=3,
                               stride=1,
                               padding=lambda _: (1, 1),
                               w_init=jnp.zeros,
                               with_bias=False)

    def __call__(self, x):
        out = sigmoid(x)
        out = self.conv1(out)
        out = sigmoid(out)
        out = self.conv2(out)
        return x + out


class PreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self):
        super(PreODE, self).__init__()
        # self.model = hk.Sequential([
        #     hk.Conv2D(output_channels=64,
        #               kernel_shape=3,
        #               stride=1,
        #               padding=lambda _: (1, 1),
        #               with_bias=False),
        # ])
        self.flatten = Flatten()

    def __call__(self, x):
        # return self.model(x)
        return self.flatten(x)


class Dynamics(hk.Module):
    """
    Dynamics of the ODENet.
    """

    def __init__(self, input_shape):
        super(Dynamics, self).__init__()
        self.input_shape = input_shape
        output_channels = input_shape[-1]
        self.conv1 = ConcatConv2D(output_channels=output_channels,
                                  kernel_shape=3,
                                  stride=1,
                                  padding=lambda _: (1, 1),
                                  with_bias=False)
        self.conv2 = ConcatConv2D(output_channels=output_channels,
                                  kernel_shape=3,
                                  stride=1,
                                  padding=lambda _: (1, 1),
                                  w_init=jnp.zeros,
                                  with_bias=False)

    def __call__(self, x, t):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, self.input_shape)

        out = sigmoid(x)
        out = self.conv1(out, t)
        out = sigmoid(out)
        out = self.conv2(out, t)

        return out


class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """

    def __init__(self, input_shape):
        super(MLPDynamics, self).__init__()
        self.input_shape = input_shape
        self.dim = jnp.prod(input_shape[1:])
        self.hidden_dim = 100
        self.lin1 = hk.Linear(self.hidden_dim,
                              with_bias=False)
        self.lin2 = hk.Linear(self.dim,
                              w_init=jnp.zeros,
                              with_bias=False)

    def __call__(self, x, t):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, (-1, self.dim))

        out = sigmoid(x)
        tt = jnp.ones_like(x[:, :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin1(t_out)

        out = sigmoid(out)
        tt = jnp.ones_like(out[:, :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin2(t_out)

        return out


class PostODE(hk.Module):
    """
    Module applied after the ODE layer.
    """

    def __init__(self):
        super(PostODE, self).__init__()
        self.model = hk.Sequential([
            # hk.AvgPool(window_shape=(1, 4, 4, 1),
            #            strides=(1, 1, 1, 1),
            #            padding="VALID"),
            # Flatten(),
            hk.Linear(10)
        ])

    def __call__(self, x):
        return self.model(x)


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


def initialization_data(input_shape, in_ode_shape, out_ode_shape):
    """
    Data for initializing the modules.
    """
    in_ode_shape = (1, ) + in_ode_shape[1:]
    in_ode_dim = jnp.prod(in_ode_shape[1:])
    out_ode_shape = (1, ) + out_ode_shape[1:]
    out_ode_dim = jnp.prod(out_ode_shape[1:])
    data = {
        "pre_ode": jnp.zeros(input_shape),
        "ode": (jnp.zeros(in_ode_dim), 0.),
        "res": jnp.zeros(in_ode_shape),
        "post_ode": jnp.zeros(out_ode_dim) if odenet or True else jnp.zeros(out_ode_shape)
    }
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (1, 28, 28, 1)
    in_ode_shape = (-1, 28, 28, 1)
    out_ode_shape = (-1, 28, 28, 1)

    initialization_data_ = initialization_data(input_shape, in_ode_shape, out_ode_shape)

    if odenet:
        pre_ode = hk.transform(wrap_module(PreODE))
        pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
        pre_ode_fn = pre_ode.apply
    else:
        pre_ode = hk.transform(wrap_module(PreODE))
        pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
        pre_ode_fn = pre_ode.apply

    if odenet:
        dynamics = hk.transform(wrap_module(MLPDynamics, in_ode_shape))
        dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
        dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)
        if reg:
            def reg_dynamics(y, t, params):
                """
                Dynamics of regularization for ODE integration.
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
                Dynamics augmented with regularization.
                """
                y, r = yr
                dydt = dynamics_wrap(y, t, params)
                drdt = reg_dynamics(y, t, params)
                return dydt, drdt
            if vmap:
                nodeint = jax.vmap(lambda y0, t, params: odeint(aug_dynamics, aug_init(y0), t, params, **ode_kwargs)[0],
                                   (0, None, None), 1)
            else:
                nodeint = lambda y0, t, params: odeint(aug_dynamics, aug_init(y0), t, params, **ode_kwargs)[0]
        else:
            if vmap:
                nodeint = jax.vmap(lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[0],
                                   (0, None, None), 1)
            else:
                nodeint = lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[0]

        def ode(params, out_pre_ode):
            """
            Apply the ODE block.
            """
            out_ode, out_ode_r = nodeint(out_pre_ode, ts, params)
            return out_ode[-1], out_ode_r[-1]

        if count_nfe:
            if vmap:
                unreg_nodeint = jax.vmap(lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1],
                                         (0, None, None))
            else:
                unreg_nodeint = lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1]

            @jax.jit
            def nfe_fn(params, _images, _labels):
                """
                Function to return NFE.
                """
                in_ode = pre_ode_fn(params["pre_ode"], _images)
                f_nfe = unreg_nodeint(in_ode, ts, params["ode"])
                return jnp.mean(f_nfe)

        else:
            nfe_fn = None

    else:
        resnet = hk.transform(wrap_module(
            lambda: hk.Sequential([ResBlock(in_ode_shape[-1]) for _ in range(num_blocks)])))
        resnet_params = resnet.init(rng, initialization_data_["res"])
        resnet_fn = resnet.apply

    post_ode = hk.transform(wrap_module(PostODE))
    post_ode_params = post_ode.init(rng, initialization_data_["post_ode"])
    post_ode_fn = post_ode.apply

    # return a dictionary of the three components of the model
    model = {
        "model": {
            "pre_ode": pre_ode_fn,
            "post_ode": post_ode_fn
        },
        "params": {
            "pre_ode": pre_ode_params,
            "post_ode": post_ode_params
        }
    }

    if odenet:
        model["model"]["ode"] = ode
        model["params"]["ode"] = dynamics_params
        model["nfe"] = nfe_fn
    else:
        model["model"]["res"] = resnet_fn
        model["params"]["res"] = resnet_params

    def forward(params, _images):
        """
        Forward pass of the model.
        """
        model_ = model["model"]

        if odenet:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode, regs = model_["ode"](params["ode"], out_pre_ode)
            logits = model_["post_ode"](params["post_ode"], out_ode)
        else:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode = model_["res"](params["res"], out_pre_ode)
            regs = jnp.zeros(_images.shape[0])
            logits = model_["post_ode"](params["post_ode"], out_ode)

        return logits, regs

    return forward, model


def aug_init(y):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    return y, jnp.zeros(y.shape[0])


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


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, images, labels):
    """
    The loss function for training.
    """
    logits, regs = forward(params, images)
    loss_ = _loss_fn(logits, labels)
    reg_ = _reg_loss_fn(regs)
    weight_ = _weight_fn(params)
    return loss_ + lam * reg_ + lam_w * weight_


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

    test_batch_size = parse_args.test_batch_size if odenet else 10000
    assert num_train % test_batch_size == 0
    num_test_batches = num_train // test_batch_size

    # make sure we always save the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0

    import tensorflow as tf

    def preprocess(img, label):
        """
        Preprocess with data augmentation.
        """
        # convert dtype
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        label = tf.cast(label, tf.int32)

        # normalize image
        img /= 255.

        return img, label

    # process the dataset
    ds_train = ds_train.map(preprocess, num_parallel_calls=10)

    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(1000)
    ds_train, ds_train_eval = ds_train.batch(parse_args.batch_size), ds_train.batch(test_batch_size)
    ds_train, ds_train_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_train_eval)

    meta = {
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

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

    forward, model = init_model()
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    def lr_schedule(train_itr):
        """
        Learning rate schedule. Implemented in lax.
        """
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 60, 1e-1, id, 0,
                        lambda _: lax.cond(_epoch < 100, 1e-2, id, 0,
                                           lambda _: lax.cond(_epoch < 140, 1e-3, id, 1e-4, id)))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)
    # opt_init, opt_update, get_params = optimizers.adam(step_size=parse_args.lr)
    opt_state = opt_init(model["params"])

    @jax.jit
    def update(_itr, _opt_state, _batch):
        """
        Update the params based on grad for current batch.
        """
        images, labels = _batch
        grad_ = grad_fn(get_params(_opt_state), images, labels)
        return opt_update(_itr, grad_, _opt_state)

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
        sep_acc_, sep_loss_aug_, sep_loss_, sep_loss_reg_, nfe = [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)

            test_batch_acc_, test_batch_loss_aug_, test_batch_loss_, test_batch_loss_reg_ = \
                sep_losses(opt_state, test_batch)

            if count_nfe:
                nfe.append(model["nfe"](get_params(opt_state), *test_batch))
            else:
                nfe.append(0)

            sep_acc_.append(test_batch_acc_)
            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_reg_.append(test_batch_loss_reg_)

        sep_acc_ = jnp.array(sep_acc_)
        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_reg_ = jnp.array(sep_loss_reg_)
        nfe = jnp.array(nfe)

        return jnp.mean(sep_acc_), jnp.mean(sep_loss_aug_), jnp.mean(sep_loss_), jnp.mean(sep_loss_reg_), jnp.mean(nfe)

    itr = 0
    info = collections.defaultdict(dict)
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)

            itr += 1

            opt_state = update(itr, opt_state, batch)

            if itr % parse_args.test_freq == 0:
                acc_, loss_aug_, loss_, loss_reg_, nfe_ = evaluate_loss(opt_state, ds_train_eval)

                print_str = 'Iter {:04d} | Accuracy {:.6f} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f} | NFE {:.6f}'.format(itr, acc_, loss_aug_, loss_, loss_reg_, nfe_)

                print(print_str)

                outfile = open("%s/reg_%s_lam_%.4e_num_blocks_%d_info.txt" % (dirname, reg, lam, num_blocks), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["acc"] = acc_
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
