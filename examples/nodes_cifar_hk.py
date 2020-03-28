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
from haiku.data_structures import to_immutable_dict
from haiku.initializers import TruncatedNormal
from jax import lax
import tree

import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.interpreters.xla import DeviceArray

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=350)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none', 'r3'], default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=5000)
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
lam_w = 1e-4
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
dirname = parse_args.dirname
odenet = False if parse_args.resnet is True else True
count_nfe = True if parse_args.count_nfe is True else False
num_blocks = parse_args.num_blocks


# some primitive functions
def sigmoid(z):
  """
  Numerically stable sigmoid.
  """
  return jnp.where(z >= 0, 1/(1 + jnp.exp(-z)), jnp.exp(z) / (1 + jnp.exp(z)))


def tanh(z):
  """
  Numerically stable tanh.
  """
  return 2 * sigmoid(2 * z) - 1


def softplus(z, threshold=20):
  """
  Numerically stable softplus.
  """
  return jax.nn.relu(z)
  # return jnp.where(z >= threshold, z, jnp.log1p(jnp.exp(z)))


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


class BottleNeckBlockV1Softplus(hk.Module):
  """
  Bottleneck Block for a ResNet implementation using softplus.
  """

  def __init__(self,
               channels,
               stride,
               use_projection,
               bn_config,
               name):
    super(BottleNeckBlockV1Softplus, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._bn_config = bn_config

    batchnorm_args = {"create_scale": True, "create_offset": True}
    batchnorm_args.update(bn_config)

    if self._use_projection:
      self._proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")
      self._proj_batchnorm = hk.BatchNorm(
          name="shortcut_batchnorm", **batchnorm_args)

    self._layers = []
    conv_0 = hk.Conv2D(
        output_channels=channels // 4,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding="SAME",
        name="conv_0")
    self._layers.append(
        [conv_0,
         hk.BatchNorm(name="batchnorm_0", **batchnorm_args)])

    conv_1 = hk.Conv2D(
        output_channels=channels // 4,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding="SAME",
        name="conv_1")
    self._layers.append(
        [conv_1,
         hk.BatchNorm(name="batchnorm_1", **batchnorm_args)])

    conv_2 = hk.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding="SAME",
        name="conv_2")
    batchnorm_2 = hk.BatchNorm(
        name="batchnorm_2", scale_init=jnp.zeros, **batchnorm_args)
    self._layers.append([conv_2, batchnorm_2])

  def __call__(self, inputs, is_training):
    if self._use_projection:
      shortcut = self._proj_conv(inputs)
      shortcut = self._proj_batchnorm(shortcut, is_training=is_training)
    else:
      shortcut = inputs

    net = inputs
    for i, [conv_layer, batchnorm_layer] in enumerate(self._layers):
      net = conv_layer(net)
      net = batchnorm_layer(net, is_training=is_training)
      net = softplus(net) if i < 2 else net  # Don't apply relu on last layer

    return softplus(net + shortcut)


class BlockGroup(hk.Module):
  """
  Higher level block for ResNet implementation using custom Bottleneck with softplus.
  """

  def __init__(self,
               channels,
               num_blocks,
               stride,
               bn_config,
               name):
    super(BlockGroup, self).__init__(name=name)
    self._channels = channels
    self._num_blocks = num_blocks
    self._stride = stride
    self._bn_config = bn_config

    self._blocks = []
    for id_block in range(num_blocks):
      self._blocks.append(
          BottleNeckBlockV1Softplus(
              channels=channels,
              stride=stride if id_block == 0 else 1,
              use_projection=(id_block == 0),
              bn_config=bn_config,
              name="block_%d" % id_block))

  def __call__(self, inputs, is_training):
    net = inputs
    for block in self._blocks:
      net = block(net, is_training=is_training)
    return net


class ResNet(hk.Module):
    """
    ResNet on CIFAR.
    Uses softplus instead of Relu for ODEs and higher-order derivatives.
    """

    def __init__(self,
                 blocks_per_group_list,
                 num_classes,
                 channels_per_group_list=(64, 128, 256, 512)):
        super(ResNet, self).__init__(name="ResNet")
        self._bn_config = {"decay_rate": 0.9, "eps": 1e-5}

        # Number of blocks in each group for ResNet.
        if len(blocks_per_group_list) != 4:
          raise ValueError(
              "`blocks_per_group_list` must be of length 4 not {}".format(
                  len(blocks_per_group_list)))
        self._blocks_per_group_list = blocks_per_group_list

        # Number of channels in each group for ResNet.
        if len(channels_per_group_list) != 4:
          raise ValueError(
              "`channels_per_group_list` must be of length 4 not {}".format(
                  len(channels_per_group_list)))
        self._channels_per_group_list = channels_per_group_list

        self._initial_conv = hk.Conv2D(output_channels=64,
                                       kernel_shape=3,
                                       stride=1,
                                       padding=lambda x: (1, 1),
                                       with_bias=False,
                                       name="initial_conv")

        self._initial_batchnorm = hk.BatchNorm(create_scale=True,
                                               create_offset=True,
                                               **self._bn_config,
                                               name="initial_batchnorm")

        self._block_groups = []
        strides = [1, 2, 2, 2]
        for i in range(4):
          self._block_groups.append(
              BlockGroup(
                  channels=self._channels_per_group_list[i],
                  num_blocks=self._blocks_per_group_list[i],
                  stride=strides[i],
                  bn_config=self._bn_config,
                  name="block_group_%d" % i))

        self._logits = hk.Linear(output_size=num_classes, w_init=jnp.zeros, name="logits")

    def __call__(self, inputs, is_training):
        net = inputs
        net = self._initial_conv(net)
        net = self._initial_batchnorm(net, is_training=is_training)
        net = softplus(net)

        for block_group in self._block_groups:
            net = block_group(net, is_training)

        net = jnp.mean(net, axis=[1, 2])

        return self._logits(net)


def loss_fn(images, labels, _odenet, _count_nfe, is_training):
    """
    The loss function for training.
    """
    model = ResNet(blocks_per_group_list=[2, 2, 2, 2],
                   num_classes=10)
    logits = model(images, is_training=is_training)
    if count_nfe:
        hk.set_state("nfe", 0)
    loss_ = _loss_fn(logits, labels)
    reg_ = _reg_loss_fn(0)
    acc_ = _acc_fn(logits, labels)
    hk.set_state("loss", loss_)
    hk.set_state("reg", reg_)
    hk.set_state("acc", acc_)
    return loss_ + lam * reg_


loss_obj = hk.transform_with_state(lambda images, labels, is_training: loss_fn(images,
                                                                               labels,
                                                                               odenet,
                                                                               count_nfe,
                                                                               is_training))


def l2_loss(params):
    """
    L2 regularization.
    """
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def loss_obj_decay(params, state, images, labels):
    """
    Top level loss objective with weight decay.
    """
    total_loss_, state = loss_obj.apply(params, state, None, images, labels, is_training=True)
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]
    l2_loss_ = lam_w * l2_loss(l2_params)
    return total_loss_ + l2_loss_


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    (ds_train, ds_test), ds_info = tfds.load('cifar10',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)
    num_train = ds_info.splits['train'].num_examples
    assert num_train % parse_args.batch_size == 0
    num_batches = num_train // parse_args.batch_size

    # make sure we always save and report the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0
    assert num_batches * parse_args.nepochs % parse_args.test_freq == 0

    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(1000)
    ds_train, ds_train_eval = ds_train.batch(parse_args.batch_size), ds_train.batch(parse_args.test_batch_size)
    ds_train, ds_train_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_train_eval)

    # initialize
    _images, _labels = next(tfds.as_numpy(ds_test.take(1)))
    _images = _images.astype(jnp.float32)
    opt_init_params, state = loss_obj.init(rng, jnp.expand_dims(_images, axis=0), _labels, is_training=True)

    def lr_schedule(train_itr):
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 150, 1e-1, id, 0, lambda x: lax.cond(_epoch < 250, 1e-2, id, 1e-3, id))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)
    opt_state = opt_init(opt_init_params)

    @jax.jit
    def update(i, opt_state, state, batch,):
        """
        Update the params based on grad for current batch.
        """
        images, labels = batch
        grad_fn = jax.grad(loss_obj_decay)
        return opt_update(i, grad_fn(get_params(opt_state), state, images, labels), opt_state)

    @jax.jit
    def sep_losses(opt_state, state, batch):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(opt_state)
        images, labels = batch
        total_loss_, state = loss_obj.apply(params, state, None, images, labels, is_training=False)
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
            test_batch = test_batch[0].astype(jnp.float32), test_batch[1]

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
            batch = batch[0].astype(jnp.float32), batch[1]
            itr += 1

            opt_state = update(itr, opt_state, state, batch)

            if itr % parse_args.test_freq == 0:
                acc_, loss_aug_, loss_, loss_reg_ = evaluate_loss(opt_state, state, ds_train_eval)

                print_str = 'Iter {:04d} | Accuracy {:.6f} | Total (Regularized) Loss {:.6f} | ' \
                            'Loss {:.6f} | r {:.6f}'.format(itr, acc_, loss_aug_, loss_, loss_reg_)

                print(print_str)
                print(print_str, file=sys.stderr)

            if itr % parse_args.save_freq == 0:
                if odenet:
                    param_filename = "%s/reg_%s_lam_%.4e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                else:
                    param_filename = "%s/reg_%s_lam_%.4e_num_blocks_%d_%d_fargs.pickle" % (dirname, reg, lam, num_blocks, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()


if __name__ == "__main__":
    run()
