"""
Create some pared down examples to see if we can take jets through them.
"""
import argparse
import collections
import os
import pickle
import sys

import haiku as hk
import numpy as onp
import tensorflow_datasets as tfds
from haiku.data_structures import to_immutable_dict
from haiku.initializers import TruncatedNormal

import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.ode import build_odeint, odeint
from jax.interpreters.xla import DeviceArray

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none', 'r3'], default='none')
parser.add_argument('--test_freq', type=int, default=6000)
parser.add_argument('--save_freq', type=int, default=6000)
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


class SkipConnection(hk.Module):
    """
    A type of Skip Connection module.
    """

    def __init__(self, func):
        super(SkipConnection, self).__init__()
        self.func = func

    def __call__(self, t_x):
        # t could correspond to time (for ODE dynamics),
        # or regularization (after the ODE block)
        t, x = t_x
        return (t, self.func(x))


class EndSkipConnection(hk.Module):
    """
    Stop doing SkipConnections.
    """

    def __init__(self):
        super(EndSkipConnection, self).__init__()

    def __call__(self, t_x):
        t, x = t_x
        return x


class ConcatConv2d(hk.Module):
    """
    Convolution with extra channel and skip connection for time
    .
    """

    def __init__(self, **kwargs):
        super(ConcatConv2d, self).__init__()
        self._layer = hk.Conv2D(**kwargs)

    def __call__(self, t_x):
        t, x = t_x
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return (t, self._layer(ttx))


def _get_shapes(params):
    """
    Recursive method for finding the shapes.
    """
    if isinstance(params, DeviceArray):
        return params.shape, params.dtype
    else:
        params_shapes = collections.defaultdict(dict)
        for key in params:
            params_shapes[key] = _get_shapes(params[key])
        return params_shapes


def get_shapes(params):
    """
    Returns DS w/ same shape as params, but with only the shapes.
    """
    return to_immutable_dict(_get_shapes(params))


def _init_params(shapes, bundle_name=""):
    """
    Recursive function to initialize params based on shapes.
    """
    params = collections.defaultdict(dict)
    for key in shapes:
        fq_name = bundle_name + "/" + key
        if isinstance(shapes[key], tuple):
            if key == "w":
                # note: initialization works for linear too
                fan_in_shape = onp.prod(shapes[key][0][:-1])
                stddev = 1. / onp.sqrt(fan_in_shape)
                init = TruncatedNormal(stddev=stddev)
            else:
                init = jnp.zeros
            # noinspection PyTypeChecker
            params[key] = hk.get_parameter(name=fq_name,
                                           shape=shapes[key][0],
                                           dtype=shapes[key][1],
                                           init=init)
        else:
            params[key] = _init_params(shapes[key], fq_name)
    return params


def init_params(shapes):
    """
    Initialize the parameters based on shapes.
    """
    return to_immutable_dict(_init_params(shapes))


def aug_init(y):
    """
    Initialize the augmented dynamics.
    """
    return jnp.concatenate((jnp.ravel(y), jnp.zeros(y.shape[0])))


def unpack_aug(yr, batch_size):
    """
    Unpack dynamics from augmentation.
    """
    return yr[:-batch_size], yr[-batch_size:]


class ODEBlock(hk.Module):
    """
    Block using Neural ODE.
    """

    def __init__(self, input_shape, reg=None, count_nfe=False):
        super(ODEBlock, self).__init__()
        self.input_shape = input_shape
        self.ode_dim = onp.prod(input_shape[1:])
        self.reg = reg
        self.count_nfe = count_nfe
        output_channels = input_shape[-1]
        model = hk.Sequential([
            SkipConnection(lambda x: jnp.reshape(x, input_shape)),
            SkipConnection(sigmoid),
            ConcatConv2d(output_channels=output_channels,
                         kernel_shape=3,
                         stride=1,
                         padding="SAME"),
            EndSkipConnection(),
            jnp.ravel
        ])
        dynamics = hk.transform(model)
        if self.count_nfe:
            self.unreg_nodeint = jax.jit(lambda y0, t, args: odeint(lambda y, t, params: dynamics.apply(params, (t, y)),
                                                                    y0, t, args)[1])
        self.ts = jnp.array([0., 1.])
        _params = dynamics.init(rng, (self.ts[0], jnp.ravel(jnp.zeros((1, *input_shape[1:])))))
        self._dynamics_shapes = get_shapes(_params)
        if reg:
            def reg_dynamics(y, t, params):
                """
                Dynamics of regularization for ODE integration.
                """
                if reg == "none":
                    return jnp.zeros(self.batch_size)
                else:
                    # do r3 regularization
                    y0, y_n = sol_recursive(lambda y, t: dynamics.apply(params, (t, y)), y, t)
                    r = jnp.reshape(y_n[-1], (-1, jnp.prod(self.input_shape[1:])))
                    return jnp.sum(r ** 2, axis=1)

            def aug_dynamics(yr, t, params):
                """
                Dynamics augmented with regularization.
                """
                y, r = unpack_aug(yr, self.batch_size)
                dydt = dynamics.apply(params, (t, y))
                drdt = reg_dynamics(y, t, params)
                return jnp.concatenate((dydt, drdt))

            self.nodeint = build_odeint(aug_dynamics)
        else:
            self.nodeint = build_odeint(lambda x, t, params: dynamics.apply(params, (t, x)))

    def __call__(self, x):
        self.batch_size = x.shape[0]
        params = init_params(self._dynamics_shapes)
        if self.count_nfe:
            self.nfe = self.unreg_nodeint(jnp.ravel(x), self.ts, params)
        if self.reg:
            y1, r1 = unpack_aug(self.nodeint(aug_init(x), self.ts, params)[-1], self.batch_size)
            return r1, jnp.reshape(y1, self.input_shape)
        else:
            return jnp.reshape(self.nodeint(jnp.ravel(x), self.ts, params)[-1], self.input_shape)


class ResBlock(hk.Module):
    """
    Standard ResBlock.
    """

    def __init__(self, output_channels):
        super(ResBlock, self).__init__()

        def residual(x_f_x):
            """
            Residual connection.
            """
            x, f_x = x_f_x
            return x + f_x
        self._model = hk.Sequential([
            lambda x: (x, x),  # copy the input for skip connections
            SkipConnection(sigmoid),
            SkipConnection(hk.Conv2D(output_channels=output_channels,
                                     kernel_shape=3,
                                     stride=1,
                                     padding="SAME")),
            residual
        ])

    def __call__(self, x):
        return self._model(x)


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


def loss_fn(images, labels):
    """
    The loss function for training.
    """
    # TODO: this shape needs to be set manually
    ode_shape = (-1, 4, 4, 8)
    if odenet:
        block = [ODEBlock(ode_shape, reg=parse_args.reg, count_nfe=count_nfe)]
    else:
        # chain resblocks and add dummy regularization at end to match API for ODENet
        block = [ResBlock(ode_shape[-1]) for _ in range(num_blocks)] + [lambda x: (0, x)]
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=8,
                  kernel_shape=3,
                  stride=2,
                  padding=lambda x: (1, 1)),
        hk.AvgPool(window_shape=(1, 11, 11, 1),
                   strides=(1, 1, 1, 1),
                   padding="VALID"),
        *block,
        SkipConnection(sigmoid),
        SkipConnection(hk.AvgPool(window_shape=(1, 3, 3, 1),
                                  strides=(1, 1, 1, 1),
                                  padding="VALID")),
        SkipConnection(Flatten()),
        SkipConnection(hk.Linear(10))
    ])
    regs, logits = model(images)
    if count_nfe:
        hk.set_state("nfe", model.layers[6].nfe)
    loss_ = _loss_fn(logits, labels)
    reg_ = _reg_loss_fn(regs)
    acc_ = _acc_fn(logits, labels)
    hk.set_state("loss", loss_)
    hk.set_state("reg", reg_)
    hk.set_state("acc", acc_)
    return loss_ + lam * reg_


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    (ds_train, ds_test), ds_info = tfds.load('mnist',
                                             split=['train', 'test'],
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

    loss_obj = hk.transform_with_state(loss_fn)

    # initialize
    _images, _labels = next(tfds.as_numpy(ds_test.take(1)))
    opt_init_params, state = loss_obj.init(rng, jnp.expand_dims(_images, axis=0), _labels)
    opt_init, opt_update, get_params = optimizers.adam(step_size=parse_args.lr)
    opt_state = opt_init(opt_init_params)

    @jax.jit
    def update(i, opt_state, state, batch,):
        """
        Update the params based on grad for current batch.
        """
        images, labels = batch
        grad_fn = lambda *args: jax.grad(loss_obj.apply, has_aux=True)(*args)[0]
        return opt_update(i, grad_fn(get_params(opt_state), state, None, images, labels), opt_state)

    @jax.jit
    def sep_losses(opt_state, state, batch):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(opt_state)
        images, labels = batch
        total_loss_, state = loss_obj.apply(params, state, None, images, labels)
        loss_ = state["~"]["loss"]
        reg_ = state["~"]["reg"]
        acc_ = state["~"]["acc"]
        return acc_, total_loss_, loss_, reg_

    def evaluate_loss(opt_state, state, ds_train_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        test_batch_size = parse_args.test_batch_size if odenet else num_train
        num_test_batches = num_train // test_batch_size
        sep_acc_, sep_loss_aug_, sep_loss_, sep_loss_reg_ = [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)

            test_batch_acc_, test_batch_loss_aug_, test_batch_loss_, test_batch_loss_reg_ = \
                sep_losses(opt_state, state, test_batch)

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
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)
            itr += 1

            opt_state = update(itr, opt_state, state, batch)

            if itr % parse_args.test_freq == 0:
                acc_, loss_aug_, loss_, loss_reg_ = evaluate_loss(opt_state, state, ds_train_eval)

                print_str = 'Iter {:04d} | Accuracy {:.6f} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f}'.format(itr, acc_, loss_aug_, loss_, loss_reg_)

                print(print_str)
                print(print_str, file=sys.stderr)

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()


if __name__ == "__main__":
    run()
