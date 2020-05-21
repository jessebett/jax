"""
FFJORD on MNIST, implemented with Haiku.
"""
import time

import haiku as hk
import tensorflow_datasets as tfds

import jax
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet
from jax.scipy.special import logit

from jax.config import config
config.update("jax_enable_x64", True)

# set up config

reg = "none"
lam = 0
lam_w = 0
seed = 0
batch_size = 200
test_batch_size = 200
rng = jax.random.PRNGKey(seed)
num_blocks = 0
ode_kwargs = {
    "atol": 1.4e-8,
    "rtol": 1.4e-8
}
softplus = lambda x: jnp.where(x >= 0,
                               x + jnp.log1p(jnp.exp(-x)),
                               jnp.log1p(jnp.exp(x)))


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

  return (jnp.reshape(y0[:-1], z_shape), [jnp.reshape(y1[:-1], z_shape)])


# set up modules
class ConcatConv2D(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, *args, **kwargs):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(*args, **kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


def logdetgrad(x, alpha):
    """
    Log determinant grad of the logit function for propagating logpx.
    """
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad_ = -jnp.log(s - s * s) + jnp.log(1 - 2 * alpha)
    return jnp.sum(jnp.reshape(logdetgrad_, (x.shape[0], -1)), axis=1, keepdims=True)


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # # normal
    # return jax.random.normal(key, shape)
    # rademacher
    return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float64) * 2 - 1


class ForwardPreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self, alpha=1e-6):
        super(ForwardPreODE, self).__init__()
        self.alpha = alpha

    def __call__(self, x, logpx):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = logit(s)
        logpy = logpx - logdetgrad(x, self.alpha)
        return y, logpy


class NN_Dynamics(hk.Module):
    """
    NN_Dynamics of the ODENet.
    """

    def __init__(self,
                 hidden_dims=(64, 64, 64),
                 input_shape=(28, 28, 1),
                 strides=(1, 1, 1, 1)):
        super(NN_Dynamics, self).__init__()
        self.input_shape = input_shape
        layers = []
        activation_fns = []
        base_layer = ConcatConv2D
        nonlinearity = softplus

        for dim_out, stride in zip(hidden_dims + (input_shape[-1],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"kernel_shape": 3, "stride": 1, "padding": lambda _: (1, 1)}
            elif stride == 2:
                layer_kwargs = {"kernel_shape": 4, "stride": 2, "padding": lambda _: (1, 1)}
            elif stride == -2:
                # note: would need to use convtranspose instead here
                layer_kwargs = {"kernel_shape": 4, "stride": 2, "padding": lambda _: (1, 1), "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(dim_out, **layer_kwargs)
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
    input_shape = (test_batch_size, ) + input_shape[1:]
    data = {
        "pre_ode": aug_init(jnp.zeros(input_shape))[:-1],
        "ode": aug_init(jnp.zeros(input_shape))[:-2] + (0., )
    }
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (-1, 28, 28, 1)

    initialization_data_ = initialization_data(input_shape)

    pre_ode = hk.transform(wrap_module(ForwardPreODE))
    pre_ode_params = pre_ode.init(rng, *initialization_data_["pre_ode"])
    pre_ode_fn = pre_ode.apply

    dynamics = hk.transform(wrap_module(NN_Dynamics))
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
    nodeint = lambda y0, ts, eps, params: odeint(aug_dynamics, y0, ts, eps, params, **ode_kwargs)

    def ode(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, rs = nodeint(reg_init(y, delta_logp), ts, eps, params)
        return ys[-1], delta_logps[-1], rs[-1]

    def forward(key, params, _images):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _images.shape)

        z, detla_logp = pre_ode_fn(params["pre_ode"], *aug_init(_images)[:-1])
        z, delta_logp, regs = ode(params["ode"], z, detla_logp, eps)

        return z, delta_logp, regs

    model = {"model": {
        "pre_ode": pre_ode_fn,
        "ode": ode
    }, "params": {
        "pre_ode": pre_ode_params,
        "ode": dynamics_params
    }
    }

    return forward, model


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

    logpx_per_dim = jnp.sum(logpx) / z.size  # averaged over batches
    bits_per_dim = -(logpx_per_dim - jnp.log(256)) / jnp.log(2)

    return bits_per_dim


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
    (ds_train, ds_test), ds_info = tfds.load('mnist',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             with_info=True,
                                             as_supervised=True,
                                             read_config=tfds.ReadConfig(shuffle_seed=seed,
                                                                         try_autocache=False))

    num_train = ds_info.splits['train'].num_examples
    num_test = ds_info.splits['test'].num_examples

    num_batches = num_train // batch_size
    num_test_batches = num_test // test_batch_size

    ds_train = ds_train.cache().repeat().shuffle(1000, seed=seed).batch(batch_size)
    ds_test_eval = ds_test.batch(test_batch_size).repeat()

    ds_train, ds_test_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_test_eval)

    meta = {
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test_eval, meta


def run(itrs):
    """
    Run the experiment.
    """

    # init the model first so that jax gets enough GPU memory before TFDS
    forward, model = init_model()
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    ds_train, ds_test_eval, meta = init_data()

    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    init_params = model["params"]
    opt_state = opt_init(init_params)

    @jax.jit
    def update(_itr, _opt_state, _key, _batch):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(_itr, grad_fn(get_params(_opt_state), _batch, _key), _opt_state)

    itr = 0

    key = rng

    for i in range(itrs):
        key, key2 = jax.random.split(key, num=2)
        batch = next(ds_train)[0]
        batch = (batch.astype(jnp.float64) + jax.random.uniform(key2,
                                                                minval=1e-15,
                                                                maxval=1 - 1e-15,
                                                                shape=batch.shape)) / 256.

        itr += 1

        update_start = time.time()
        opt_state = update(itr, opt_state, key, batch)
        tree_flatten(opt_state)[0][0].block_until_ready()
        update_end = time.time()
        time_str = "%d %.18f" % (itr, update_end - update_start)
        print(time_str)
