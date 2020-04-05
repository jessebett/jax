"""
Neural ODEs on CIFAR, implemented with Haiku.
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
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1e-7)
parser.add_argument('--rtol', type=float, default=1e-7)
parser.add_argument('--method', type=str, default="dopri5")
parser.add_argument('--no_vmap', action="store_true")
parser.add_argument('--init_step', type=float, default=-1.)
parser.add_argument('--reg', type=str, choices=['none', 'r3'], default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=5000)
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
# TODO: implement primitives to allow numerical stability
def sigmoid(z):
  """
  Numerically stable sigmoid.
  """
  return 1/(1 + jnp.exp(-z))
  # return jnp.where(z >= 0, 1/(1 + jnp.exp(-z)), jnp.exp(z) / (1 + jnp.exp(z)))


# TODO: fix this
def softplus(z, threshold=20):
  """
  Numerically stable softplus.
  """
  return jnp.where(z >= threshold, z, jnp.log1p(jnp.exp(z)))


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


class ConcatConv2d(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, **kwargs):
        super(ConcatConv2d, self).__init__()
        self._layer = hk.Conv2D(**kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


class LayerNorm(hk.Module):
    """
    Layer normalization.
    """

    def __init__(self, ):
        super(LayerNorm, self).__init__()
        # TODO: need to make a new general GroupNorm module
        self._layer = hk.BatchNorm(create_scale=True,
                                   create_offset=True,
                                   axis=[1, 2, 3])

    def __call__(self, x):
        # for Layer Norm we always use test-time statistics
        return self._layer(inputs=x,
                           test_local_stats=True,
                           is_training=True)


class ResBlock(hk.Module):
    """
    Standard ResBlock.
    """
    expansion = 1

    def __init__(self, in_channels, output_channels, bn_config, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = hk.BatchNorm(**bn_config)
        self.conv1 = hk.Conv2D(output_channels=output_channels,
                               kernel_shape=3,
                               stride=stride,
                               padding=lambda _: (1, 1),
                               with_bias=False)
        self.bn2 = hk.BatchNorm(**bn_config)
        self.conv2 = hk.Conv2D(output_channels=output_channels,
                               kernel_shape=3,
                               stride=1,
                               padding=lambda _: (1, 1),
                               w_init=jnp.zeros,
                               with_bias=False)

        if stride != 1 or in_channels != self.expansion * output_channels:
            self.has_shortcut = True
            self.shortcut = hk.Sequential([
                hk.Conv2D(output_channels=self.expansion * output_channels,
                          kernel_shape=1,
                          stride=stride,
                          padding="VALID",
                          with_bias=False)
            ])
        else:
            self.has_shortcut = False

    def __call__(self, x, is_training):
        out = self.bn1(x, is_training=is_training)
        out = jax.nn.relu(out)
        shortcut = self.shortcut(out) if self.has_shortcut else x
        out = self.conv1(out)
        out = self.bn2(out, is_training=is_training)
        out = jax.nn.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class ResNet(hk.Module):
    """
    For chaining together residual blocks.
    hk.Sequential doesn't work since it doesn't allow is_training.
    """
    def __init__(self,
                 block_nums,
                 bn_config=None,
                 channels_per_group=(64, 128, 256, 512)):
        super(ResNet, self).__init__()
        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        assert len(block_nums) == 4
        assert len(channels_per_group) == 4

        self._block_groups = []
        strides = (1, 2, 2, 2)
        for i in range(4):
            block_group = []
            for j in range(block_nums[i]):
                block_group.append(ResBlock(in_channels=64,
                                            output_channels=channels_per_group[i],
                                            bn_config=bn_config,
                                            stride=(1 if j else strides[i])))
            self._block_groups.append(block_group)

    def __call__(self, x, is_training):
        # for block_group in self._block_groups:
        #     for block in block_group:
        #         x = block(x, is_training)

        return x


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


class PreODE(hk.Module):
    """
    Module applied before the ODE layer.
    From pre-activation resnet.
    """
    def __init__(self):
        super(PreODE, self).__init__()
        self.model = hk.Sequential([
            hk.Conv2D(output_channels=64,
                      kernel_shape=3,
                      stride=1,
                      padding=lambda _: (1, 1),
                      with_bias=False),
        ])

    def __call__(self, x):
        return self.model(x)


class PreODERes(hk.Module):
    """
    PreODE and beginngin of resnet.
    """
    def __init__(self,
                 block_nums,
                 bn_config=None,
                 channels_per_group=(64, 128, 256)):
        super(PreODERes, self).__init__()
        self.model = PreODE()
        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        assert len(block_nums) == 3
        assert len(channels_per_group) == 3

        self._block_groups = []
        strides = (1, 2, 2)
        for i in range(3):
            block_group = []
            for j in range(block_nums[i]):
                block_group.append(ResBlock(in_channels=64,
                                            output_channels=channels_per_group[i],
                                            bn_config=bn_config,
                                            stride=(1 if j else strides[i])))
            self._block_groups.append(block_group)

    def __call__(self, x, is_training):
        x = self.model(x)
        for block_group in self._block_groups:
            for block in block_group:
                x = block(x, is_training)
        return x


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


class PostODE(hk.Module):
    """
    Module applied after the ODE layer.
    """

    def __init__(self):
        super(PostODE, self).__init__()
        self.model = hk.Sequential([
            hk.AvgPool(window_shape=(1, 4, 4, 1),
                       strides=(1, 1, 1, 1),
                       padding="VALID"),
            Flatten(),
            hk.Linear(10)
        ])

    def __call__(self, x):
        return self.model(x)


class PostODERes(hk.Module):
    """
    Module applied after the ODE layer, including last bit of resnet.
    """
    def __init__(self,
                 block_nums,
                 bn_config=None,
                 channels_per_group=(512, )):
        super(PostODERes, self).__init__()
        self.model = PostODE()
        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        assert len(block_nums) == 1
        assert len(channels_per_group) == 1

        self._block_groups = []
        strides = (2, )
        for i in range(1):
            block_group = []
            for j in range(block_nums[i]):
                block_group.append(ResBlock(in_channels=64,
                                            output_channels=channels_per_group[i],
                                            bn_config=bn_config,
                                            stride=(1 if j else strides[i])))
            self._block_groups.append(block_group)

    def __call__(self, x, is_training):
        for block_group in self._block_groups:
            for block in block_group:
                x = block(x, is_training)
        x = self.model(x)
        return x


def initialization_data(input_shape, in_ode_shape, out_ode_shape):
    """
    Data for initializing the modules.
    """
    in_ode_shape = (1, ) + in_ode_shape[1:]
    out_ode_shape = (1, ) + out_ode_shape[1:]
    data = {
        "pre_ode": jnp.zeros(input_shape),
        "ode": (jnp.zeros(in_ode_shape), 0.),
        "res": jnp.zeros(in_ode_shape),
        "post_ode": jnp.zeros(in_ode_shape) if odenet or True else jnp.zeros(out_ode_shape)
    }
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (1, 32, 32, 3)
    in_ode_shape = (-1, 8, 8, 256)
    out_ode_shape = (-1, 4, 4, 512)

    initialization_data_ = initialization_data(input_shape, in_ode_shape, out_ode_shape)

    if odenet:
        pre_ode = hk.transform_with_state(wrap_module(PreODERes, [2, 2, 1]))
        pre_ode_params, pre_ode_state = pre_ode.init(rng, initialization_data_["pre_ode"], is_training=True)
        pre_ode_fn = lambda params, state, x, is_training: pre_ode.apply(params, state, None, x, is_training=is_training)
    else:
        pre_ode = hk.transform(wrap_module(PreODE))
        pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
        pre_ode_fn = pre_ode.apply

    if odenet:
        # TODO: how to do analagous multiple blocks
        dynamics = hk.transform(wrap_module(Dynamics, in_ode_shape))
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
                nodeint = jax.vmap(lambda y0, t, params: odeint(aug_dynamics, y0, t, params, **ode_kwargs)[0],
                                   (0, None, None), 1)
            else:
                nodeint = lambda y0, t, params: odeint(aug_dynamics, y0, t, params, **ode_kwargs)[0]
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
            out_ode, out_ode_r = nodeint(aug_init(out_pre_ode), ts, params)
            return out_ode[-1], out_ode_r[-1]

        if count_nfe:
            if vmap:
                unreg_nodeint = jax.vmap(lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1],
                                         (0, None, None))
            else:
                unreg_nodeint = lambda y0, t, params: odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1]

            @jax.jit
            def nfe_fn(params, state, _images, _labels):
                """
                Function to return NFE.
                """
                in_ode, _ = pre_ode_fn(params["pre_ode"], state["pre_ode"], _images, is_training=False)
                f_nfe = unreg_nodeint(in_ode, ts, params["ode"])
                return jnp.mean(f_nfe)

        else:
            nfe_fn = None

    else:
        # instantiate series of resblocks
        block_nums = [2, 2, 2, 2]
        # TODO: pass in type of resblock?
        resnet = hk.transform_with_state(wrap_module(ResNet, block_nums))
        resnet_params, resnet_state = resnet.init(rng, initialization_data_["res"], is_training=True)
        resnet_fn = lambda params, state, x, is_training: resnet.apply(params, state, None, x, is_training=is_training)

    if odenet:
        post_ode = hk.transform_with_state(wrap_module(PostODERes, [2, ]))
        post_ode_params, post_ode_state = post_ode.init(rng, initialization_data_["post_ode"], is_training=True)
        post_ode_fn = lambda params, state, x, is_training: post_ode.apply(params, state, None, x, is_training=is_training)
    else:
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
        },
        "state": {
            "pre_ode": pre_ode_state if odenet else None,
            "resnet": None if odenet else resnet_state,
            "post_ode": post_ode_state if odenet else None
        }
    }

    if odenet:
        model["model"]["ode"] = ode
        model["params"]["ode"] = dynamics_params
        model["nfe"] = nfe_fn
    else:
        model["model"]["res"] = resnet_fn
        model["params"]["res"] = resnet_params

    def forward(params, state, _images, is_training):
        """
        Forward pass of the model.
        """
        model_ = model["model"]

        new_state = collections.defaultdict(lambda: None)
        if odenet:
            out_pre_ode, new_state["pre_ode"] = \
                model_["pre_ode"](params["pre_ode"], state["pre_ode"], _images, is_training=is_training)
            out_ode, regs = model_["ode"](params["ode"], out_pre_ode)
            logits, new_state["post_ode"] = \
                model_["post_ode"](params["post_ode"], state["post_ode"], out_ode, is_training=is_training)
        else:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode, new_state["res"] = model_["res"](params["res"], state["res"], out_pre_ode, is_training=is_training)
            regs = jnp.zeros(_images.shape[0])
            logits = model_["post_ode"](params["post_ode"], out_ode)

        return logits, regs, new_state

    return forward, model


def loss_fn(forward, params, state, images, labels, is_training):
    """
    The loss function for training.
    """
    logits, regs, state = forward(params, state, images, is_training=is_training)
    loss_ = _loss_fn(logits, labels)
    reg_ = _reg_loss_fn(regs)
    weight_ = _weight_fn(params)
    return loss_ + lam * reg_ + lam_w * weight_, state


def init_data():
    """
    Initialize data.
    """
    (ds_train,), ds_info = tfds.load('cifar10',
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
        # random crop
        img = tf.image.resize_with_crop_or_pad(img, 36, 36)
        img = tf.image.random_crop(img, [32, 32, 3])

        # random flip left and right
        img = tf.image.random_flip_left_right(img)

        # convert dtype
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        label = tf.cast(label, tf.int32)

        # normalize image
        def _normalize_image(image, mean, stddev):
            """Normalize the image to zero mean and unit variance."""
            image -= tf.constant(mean, shape=[1, 1, 3], dtype=img.dtype)
            image /= tf.constant(stddev, shape=[1, 1, 3], dtype=img.dtype)
            return image
        img = _normalize_image(img,
                               mean=(0.4914, 0.4822, 0.4465),
                               stddev=(0.2023, 0.1994, 0.2010))

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
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args, is_training=True), has_aux=True)

    def lr_schedule(train_itr):
        """
        Learning rate schedule. Implemented in lax.
        """
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 75, 1e-1, id, 0,
                        lambda _: lax.cond(_epoch < 125, 1e-2, id, 0,
                                           lambda _: lax.cond(_epoch < 175, 1e-3, id, 1e-3, id)))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)
    opt_state = opt_init(model["params"])
    state = model["state"]

    @jax.jit
    def update(_itr, _opt_state, _state, _batch):
        """
        Update the params based on grad for current batch.
        """
        images, labels = _batch
        grad_, _state = grad_fn(get_params(_opt_state), _state, images, labels)
        return opt_update(_itr, grad_, _opt_state), _state

    @jax.jit
    def sep_losses(_opt_state, _state, _batch):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        images, labels = _batch
        logits, regs, _ = forward(params, _state, images, is_training=False)
        loss_ = _loss_fn(logits, labels)
        reg_ = _reg_loss_fn(regs)
        total_loss_ = loss_ + lam * reg_
        acc_ = _acc_fn(logits, labels)
        return acc_, total_loss_, loss_, reg_

    def evaluate_loss(opt_state, state, ds_train_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_acc_, sep_loss_aug_, sep_loss_, sep_loss_reg_, nfe = [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)

            test_batch_acc_, test_batch_loss_aug_, test_batch_loss_, test_batch_loss_reg_ = \
                sep_losses(opt_state, state, test_batch)

            if count_nfe:
                nfe.append(model["nfe"](get_params(opt_state), state, *test_batch))
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

            opt_state, state = update(itr, opt_state, state, batch)

            if itr % parse_args.test_freq == 0:
                acc_, loss_aug_, loss_, loss_reg_, nfe_ = evaluate_loss(opt_state, state, ds_train_eval)

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
    meta = {
        "info": info,
        "args": parse_args
    }
    outfile = open("%s/reg_%s_lam_%.4e_num_blocks_%d_meta.pickle" % (dirname, reg, lam, num_blocks), "wb")
    pickle.dump(meta, outfile)
    outfile.close()


if __name__ == "__main__":
    run()
