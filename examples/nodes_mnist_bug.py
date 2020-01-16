"""
Neural ODEs on MNIST (only parameters are dynamics).
"""
import jax.numpy as np
from jax.examples import datasets
from jax import random, grad
from jax.experimental.ode import build_odeint
from jax.flatten_util import ravel_pytree
from jax.nn import log_softmax
from jax.nn.initializers import glorot_normal, normal

img_dim = 784
ode_dim = 200
num_reg = 1
n_classes = 10
batch_size = 128

rng = random.PRNGKey(0)

# set up data
t = np.array([0., 1.])

# set up dims to select
ode_in_dims = random.shuffle(rng, np.arange(img_dim))[:ode_dim]


def append_time(y, t):
  """
  y::(BS,D)
  t::scalar
  yt::(BS,D+1)
  """
  yt = np.concatenate((y,
                       np.ones((y.shape[0], 1)) * t),
                      axis=1)
  return yt


# set up model
def sigmoid(z):
    """
    Defined using only numpy primitives (but probably less numerically stable).
    """
    return 1./(1. + np.exp(-z))


def dynamics_predict(params, y_t):
    """
    MLP for dynamics of ODE.
    """
    w_dyn, b_dyn = params
    return sigmoid(np.dot(y_t, w_dyn) + np.expand_dims(b_dyn, axis=0))


def pre_ode(x):
    """
    Select subset of dims for ODE (since hard to integrate high-dims).
    """
    return x[:, ode_in_dims]


def post_ode(out_ode):
    """
    Select subset of output dims for classification prediction.
    """
    return log_softmax(out_ode[:, :10])


# set up ODE
def dynamics(flat_y, t, *flat_params):
    """
    Dynamics of the ODEBlock.
    """
    y = np.reshape(flat_y, (-1, ode_dim))
    dydt = dynamics_predict(ravel_ode_params(np.array(flat_params)), append_time(y, t))
    return np.ravel(dydt)


nodeint = build_odeint(dynamics)


def predict(args, x):
    """
    The prediction function for our model.
    """
    out_1 = pre_ode(x)

    flat_out_ode = nodeint(np.ravel(out_1), t, *args)[-1]
    out_ode = np.reshape(flat_out_ode, (-1, ode_dim))

    out = post_ode(out_ode)

    return out


def loss_fun(preds, targets):
    """
    Negative log-likelihood.
    """
    return -np.mean(np.sum(preds * targets, axis=1))


def loss(params, batch):
    """
    Unaugmented loss to do GD over.
    """
    inputs, targets = batch
    preds = predict(params, inputs)
    return loss_fun(preds, targets)


def initialize(rng):
    """
    Initialize parameters of the model.
    """
    rng, layer_rng = random.split(rng)
    k1, k2 = random.split(layer_rng)
    w_dyn, b_dyn = glorot_normal()(k1, (ode_dim + 1, ode_dim)), normal()(k2, (ode_dim,))

    flat_ode_params, ravel_ode_params = ravel_pytree((w_dyn, b_dyn))

    return rng, flat_ode_params, ravel_ode_params


# define ravel
_, _init_params, ravel_ode_params = initialize(rng)

train_images, train_labels, test_images, test_labels = datasets.mnist()

loss(_init_params, (train_images[:batch_size], train_labels[:batch_size]))
print("forward worked")

grad_ = grad(loss)
print("built grad")

grad_(_init_params, (train_images[:batch_size], train_labels[:batch_size]))
print("backward worked")
