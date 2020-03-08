"""
Create some pared down examples to see if we can take jets through them.
"""
import time
import collections

import haiku as hk
from haiku.initializers import TruncatedNormal
import jax
import jax.numpy as jnp
import numpy as onp
from jax.experimental.ode import build_odeint, odeint

from jax.flatten_util import ravel_pytree

from jax.interpreters.xla import DeviceArray
from haiku.data_structures import to_immutable_dict

import tensorflow_datasets as tfds
from functools import reduce

from scipy.special import factorial as fact


def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)
  return rfun


def jvp_taylor(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  n_derivs = []
  N = len(series[0]) + 1
  for i in range(1, N):
    d = repeated(jax.jacobian, i)(expansion)(0.)
    n_derivs.append(d)
  return f(*primals), n_derivs


def jvp_test_jet(f, primals, series, atol=1e-5):
  tic = time.time()
  y, terms = jax.jet(f, primals, series)
  print("jet done in {} sec".format(time.time() - tic))

  tic = time.time()
  y_jvp, terms_jvp = jvp_taylor(f, primals, series)
  print("jvp done in {} sec".format(time.time() - tic))

  assert jnp.allclose(y, y_jvp)
  assert jnp.allclose(terms, terms_jvp, atol=atol)


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


rng = jax.random.PRNGKey(42)

order = 4

batch_size = 2
train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
train_ds = train_ds.cache().shuffle(1000).batch(batch_size)
batch = next(tfds.as_numpy(train_ds))


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
    # TODO: the index is dependent on the batch size!
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
                         padding=lambda x: (1, 1)),
            SkipConnection(sigmoid),
            ConcatConv2d(output_channels=output_channels,
                         kernel_shape=3,
                         stride=1,
                         padding=lambda x: (1, 1)),
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
                    return jnp.zeros_like(y)
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


class SmallODEBlock(hk.Module):
    """
    Block using Neural ODE.
    """

    def __init__(self, input_shape, reg=None):
        super(SmallODEBlock, self).__init__()
        self.input_shape = input_shape
        self.ode_dim = onp.prod(input_shape[1:])
        self.reg = reg
        output_channels = input_shape[-1]
        model = hk.Sequential([
            SkipConnection(lambda x: jnp.reshape(x, input_shape)),
            SkipConnection(sigmoid),
            ConcatConv2d(output_channels=output_channels,
                         kernel_shape=2,
                         stride=1,
                         padding="SAME"),
            EndSkipConnection(),
            jnp.ravel
        ])
        dynamics = hk.transform(model)
        self.ts = jnp.array([0., 1.])
        _params = dynamics.init(rng, (self.ts[0], jnp.ravel(jnp.zeros((1, *input_shape[1:])))))
        self._dynamics_shapes = get_shapes(_params)
        if reg:
            def reg_dynamics(y, t, params):
                """
                Dynamics of regularization for ODE integration.
                """
                if reg == "none":
                    return jnp.zeros_like(y)
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
        if self.reg:
            y1, r1 = unpack_aug(self.nodeint(aug_init(x), self.ts, params)[-1], self.batch_size)
            return r1, jnp.reshape(y1, self.input_shape)
        else:
            return jnp.reshape(self.nodeint(jnp.ravel(x), self.ts, params)[-1], self.input_shape)


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


def test_node():
  """
  Test taking grad through ODENet.
  """

  def loss_fn(images, labels):
    # TODO: this shape needs to be set manually
    ode_shape = (-1, 6, 6, 64)
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=64,
                  kernel_shape=3,
                  stride=1,
                  padding="VALID"),
        sigmoid,
        hk.Conv2D(output_channels=64,
                  kernel_shape=4,
                  stride=2,
                  padding=lambda x: (1, 1)),
        sigmoid,
        hk.Conv2D(output_channels=64,
                  kernel_shape=4,
                  stride=2,
                  padding=lambda x: (1, 1)),
        ODEBlock(ode_shape),
        sigmoid,
        hk.AvgPool(window_shape=(1, 6, 6, 1),
                   strides=(1, 1, 1, 1),
                   padding="VALID"),
        Flatten(),
        hk.Linear(10)
    ])
    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

  loss_obj = hk.transform(loss_fn)

  images, labels = batch['image'], batch['label']

  params = loss_obj.init(rng, images, labels)

  loss_obj.apply(params, images, labels)
  print("forward pass works")

  jax.grad(loss_obj.apply)(params, images, labels)
  print("gradient works")


def test_node_small():
  """
  Test taking grad through small ODENet (for numerical diff).
  """

  def loss_fn(images, labels):
    # TODO: this shape needs to be set manually
    ode_shape = (-1, 4, 4, 2)
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=ode_shape[-1],
                  kernel_shape=2,
                  stride=1,
                  padding="VALID"),
        hk.AvgPool(window_shape=(1, 24, 24, 1),
                   strides=(1, 1, 1, 1),
                   padding="VALID"),
        SmallODEBlock(ode_shape),
        sigmoid,
        hk.AvgPool(window_shape=(1, 4, 4, 2),
                   strides=(1, 1, 1, 1),
                   padding="VALID"),
        Flatten(),
        hk.Linear(10)
    ])
    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

  loss_obj = hk.transform(loss_fn)

  images, labels = batch['image'], batch['label']

  params = loss_obj.init(rng, images, labels)

  loss_obj.apply(params, images, labels)
  print("forward pass works")

  dr2dp = jax.grad(loss_obj.apply)(params, images, labels)
  print("gradient works")

  def nd(f, x, eps=1e-5):
    """
    Numerical differentiation
    """
    flat_x, unravel = ravel_pytree(x)
    dim = len(flat_x)
    g = onp.zeros_like(flat_x)
    for i in range(dim):
      print("%d of %d" % (i + 1, dim))
      d = onp.zeros_like(flat_x)
      d[i] = eps
      g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
    return g

  dr2dp_numerical = nd(lambda params: loss_obj.apply(params, images, labels), params)

  dr2dp, _ = ravel_pytree(dr2dp)
  abs_diff = onp.abs(dr2dp - dr2dp_numerical)
  denom = onp.abs(dr2dp) + onp.abs(dr2dp_numerical)
  rel_diff = abs_diff / denom
  print("max abs. diff", onp.max(abs_diff))
  print("mean abs. diff", onp.mean(abs_diff))
  print("median abs. diff", onp.median(abs_diff))
  print(abs_diff)
  print("max rel. diff", onp.max(rel_diff))
  print("mean rel. diff", onp.mean(rel_diff))
  print("median rel. diff", onp.median(rel_diff))
  print(rel_diff)


def test_node_reg():
  """
  Test taking grad through ODENet regularized with jets.
  """
  lam = 1

  def _loss_fn(logits, labels):
      return jnp.mean(softmax_cross_entropy(logits, labels))

  def _reg_loss_fn(reg):
      return jnp.mean(reg)

  def loss_fn(images, labels):
    # TODO: this shape needs to be set manually
    ode_shape = (-1, 6, 6, 64)
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=64,
                  kernel_shape=3,
                  stride=1,
                  padding="VALID"),
        sigmoid,
        hk.Conv2D(output_channels=64,
                  kernel_shape=4,
                  stride=2,
                  padding=lambda x: (1, 1)),
        sigmoid,
        hk.Conv2D(output_channels=64,
                  kernel_shape=4,
                  stride=2,
                  padding=lambda x: (1, 1)),
        ODEBlock(ode_shape, reg="r3"),
        SkipConnection(sigmoid),
        SkipConnection(hk.AvgPool(window_shape=(1, 6, 6, 1),
                                  strides=(1, 1, 1, 1),
                                  padding="VALID")),
        SkipConnection(Flatten()),
        SkipConnection(hk.Linear(10))
    ])
    reg, logits = model(images)
    return _loss_fn(logits, labels) + lam * _reg_loss_fn(reg)

  loss_obj = hk.transform(loss_fn)

  images, labels = batch['image'], batch['label']

  params = loss_obj.init(rng, images, labels)

  loss_obj.apply(params, images, labels)
  print("forward pass works")

  jax.grad(loss_obj.apply)(params, images, labels)
  print("gradient works")


def test_node_reg_small():
  """
  Test taking grad through ODENet regularized with jets.
  """
  lam = 1

  def _loss_fn(logits, labels):
      return jnp.mean(softmax_cross_entropy(logits, labels))

  def _reg_loss_fn(reg):
      return jnp.mean(reg)

  def loss_fn(images, labels):
    # TODO: this shape needs to be set manually
    ode_shape = (-1, 4, 4, 2)
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=ode_shape[-1],
                  kernel_shape=2,
                  stride=1,
                  padding="VALID"),
        hk.AvgPool(window_shape=(1, 24, 24, 1),
                   strides=(1, 1, 1, 1),
                   padding="VALID"),
        SmallODEBlock(ode_shape, reg="r3"),
        SkipConnection(sigmoid),
        SkipConnection(hk.AvgPool(window_shape=(1, 4, 4, 2),
                                  strides=(1, 1, 1, 1),
                                  padding="VALID")),
        SkipConnection(Flatten()),
        SkipConnection(hk.Linear(10))
    ])
    reg, logits = model(images)
    return _loss_fn(logits, labels) + lam * _reg_loss_fn(reg)

  loss_obj = hk.transform(loss_fn)

  images, labels = batch['image'], batch['label']

  params = loss_obj.init(rng, images, labels)

  loss_obj.apply(params, images, labels)
  print("forward pass works")

  dr2dp = jax.grad(loss_obj.apply)(params, images, labels)
  print("gradient works")

  def nd(f, x, eps=1e-5):
    """
    Numerical differentiation
    """
    flat_x, unravel = ravel_pytree(x)
    dim = len(flat_x)
    g = onp.zeros_like(flat_x)
    for i in range(dim):
      print("%d of %d" % (i + 1, dim))
      d = onp.zeros_like(flat_x)
      d[i] = eps
      g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
    return g

  dr2dp_numerical = nd(lambda params: loss_obj.apply(params, images, labels), params)

  dr2dp, _ = ravel_pytree(dr2dp)
  abs_diff = onp.abs(dr2dp - dr2dp_numerical)
  denom = onp.abs(dr2dp) + onp.abs(dr2dp_numerical)
  rel_diff = abs_diff / denom
  print("max abs. diff", onp.max(abs_diff))
  print("mean abs. diff", onp.mean(abs_diff))
  print("median abs. diff", onp.median(abs_diff))
  print(abs_diff)
  print("max rel. diff", onp.max(rel_diff))
  print("mean rel. diff", onp.mean(rel_diff))
  print("median rel. diff", onp.median(rel_diff))
  print(rel_diff)


def test_node_reg_nfe():
  """
  Test taking grad through regularized ODENet counting NFE.
  """

  lam = 1

  def _loss_fn(logits, labels):
      return jnp.mean(softmax_cross_entropy(logits, labels))

  def _reg_loss_fn(reg):
      return jnp.mean(reg)

  def loss_fn(images, labels):
    # TODO: this shape needs to be set manually
    ode_shape = (-1, 6, 6, 64)
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=64,
                  kernel_shape=3,
                  stride=1,
                  padding="VALID"),
        sigmoid,
        hk.Conv2D(output_channels=64,
                  kernel_shape=4,
                  stride=2,
                  padding=lambda x: (1, 1)),
        sigmoid,
        hk.Conv2D(output_channels=64,
                  kernel_shape=4,
                  stride=2,
                  padding=lambda x: (1, 1)),
        ODEBlock(ode_shape, reg="r3", count_nfe=True),
        SkipConnection(sigmoid),
        SkipConnection(hk.AvgPool(window_shape=(1, 6, 6, 1),
                                  strides=(1, 1, 1, 1),
                                  padding="VALID")),
        SkipConnection(Flatten()),
        SkipConnection(hk.Linear(10))
    ])
    reg, logits = model(images)
    hk.set_state("nfe", model.layers[6].nfe)
    return _loss_fn(logits, labels) + lam * _reg_loss_fn(reg)

  loss_obj = hk.transform_with_state(loss_fn)

  images, labels = batch['image'], batch['label']

  params, state = loss_obj.init(rng, images, labels)

  loss_, state = loss_obj.apply(params, state, rng, images, labels)
  print(state['~']['nfe'])
  print("forward pass works")

  jax.grad(loss_obj.apply, has_aux=True)(params, state, rng, images, labels)
  print("gradient works")


# test_node()
# test_node_small()
# test_node_reg()
# test_node_reg_small()
test_node_reg_nfe()
