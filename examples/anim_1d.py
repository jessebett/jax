"""
Neural ODEs in Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as np
from jax import random, grad, jet
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.ode import odeint, build_odeint, vjp_odeint
from jax.flatten_util import ravel_pytree
from jax.nn.initializers import glorot_normal, normal
from scipy.special import factorial as fact

REGS = ['r0', 'r1', 'r2']
NUM_REGS = len(REGS)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--nepochs', type=int, default=1000)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--dirname', type=str, default='tmp15')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parse_args = parser.parse_args()


config.update('jax_enable_x64', True)


### TOY DATA ###
D = 1
true_y0_range = 3
true_fn = lambda x: x ** 3
# only for evaluating on a fixed test set
true_y0 = np.repeat(np.expand_dims(np.linspace(-3, 3, parse_args.data_size), axis=1), D, axis=1)  # (N, D)
true_y1 = np.repeat(np.expand_dims(true_fn(true_y0[:, 0]), axis=1), D, axis=1)
true_y = np.concatenate((np.expand_dims(true_y0, axis=0),
                        np.expand_dims(true_y1, axis=0)),
                        axis=0)  # (T, N, D)
t = np.array([0., 1.])  # (T)

@jax.jit
def get_batch(key):
    """
    Get batch.
    """
    new_key, subkey = random.split(key)
    key = new_key
    batch_y0 = random.uniform(subkey, (parse_args.batch_size, D),
                              minval=-true_y0_range, maxval=true_y0_range)              # (M, D)
    batch_y1 = np.repeat(np.expand_dims(true_fn(batch_y0[:, 0]), axis=1), D, axis=1)    # (M, D)
    batch_t = t                                                                         # (T)
    batch_y = np.concatenate((np.expand_dims(batch_y0, axis=0),
                              np.expand_dims(batch_y1, axis=0)),
                             axis=0)                                                    # (T, M, D)
    return key, batch_y0, batch_t, batch_y

@jax.jit
def pack_batch(key):
    """
    Get batch and package it for augmented system integration.
    """
    key, batch_y0, batch_t, batch_y = get_batch(key)

    batch_y0, batch_y1 = batch_y
    # batch_y0_t = np.concatenate((batch_y0,np.expand_dims(np.repeat(batch_t[0],parse_args.batch_size),axis=1)),axis=1)
    # batch_y1_t = np.concatenate((batch_y1,np.expand_dims(np.repeat(batch_t[1],parse_args.batch_size),axis=1)),axis=1)
    #
    # batch_y_t = np.array((batch_y0_t, batch_y1_t)) #(T, BS, D+1)
    return key, batch_y #(T, BS, D)

def append_time(y,t):
  """
  y::(BS,D)
  t::scalar
  yt::(BS,D+1)
  """
  yt = np.concatenate((y,np.expand_dims(np.repeat(t, parse_args.batch_size),axis=1)),axis=1)
  return yt


### MODEL ###
def sigmoid(z):
    return 1./(1. + np.exp(-z))

def predict(params, y_t):
  w_1, b_1 = params[0]
  z_1 = np.dot(y_t, w_1) + np.expand_dims(b_1, axis=0)
  out_1 = sigmoid(z_1)

  w_2, b_2 = params[1]
  z_1 = np.dot(out_1, w_2) + np.expand_dims(b_2, axis=0)

  return z_1

def jet_wrap_predict(params):
  """
  Function which returns a closure that has the correct API for jet.
  """
  return lambda z, t: predict(params, append_time(z,t))

def sol_recursive(f,z,t):
# closure to expand z
  def g(z):
      return f(z,t)

  # First coeffs computed recursively
  (y0, [y1h]) = jet(g, (z, ), ((np.ones_like(z), ), ))
  (y0, [y1, y2h]) = jet(g, (z, ), (( y0, y1h,), ))
  (y0, [y1, y2, y3h]) = jet(g, (z, ), ((y0, y1, y2h), ))
  (y0, [y1, y2, y3, y4h]) = jet(g, (z, ), ((y0, y1, y2, y3h), ))
  (y0, [y1, y2, y3, y4, y5h]) = jet(g, (z, ),
                                    ((y0, y1, y2, y3, y4h), ))
  (y0, [y1, y2, y3, y4, y5,y6h]) = jet(g, (z, ),
                                    ((y0, y1, y2, y3, y4, y5h), ))


  return (y0, [y1, y2, y3, y4, y5])
  # return  (y0, [y1, y2])

# Neural ODE
@jax.jit
def dynamics(y,t,*args):
  """
  y::unravel((BS,D))
  t::scalar
  args::flat_params
  """
  y = ravel_y(y) # because jax needs flat y
  yt = append_time(y,t)
  dydt = predict(ravel_params(np.array(args)),yt)
  return np.ravel(dydt) # jax needs flat y



@jax.jit
def loss_fun(pred, target):
    """
    Mean squared error.
    """
    return np.mean((pred - target) ** 2)

@jax.jit
def nodes_predict_and_loss(true_ys, y0, ts, flat_params):
    """
    Return function loss(params) for gradients
    """
    # true_ys, odeint_args = args[0], args[1:]
    ys_pred = nodeint(y0,ts,*flat_params)
    ys_pred = ravel_y(ys_pred[-1])
    return loss_fun(true_ys,ys_pred)

# @jax.jit
# def nodes_loss(b_y1,b_y0,params):
#    # = args
#   return nodes_predict(b_y1,np.ravel(b_y0),np.array([0.,1.]), params)

# Main:
rng = random.PRNGKey(0)
t0 = 0.
# ts = np.linspace(0.,1.,100)
ts = np.array([0.,1.])

# initialize the parameters
hidden_dim = 50
rng, layer_rng = random.split(rng)
k1, k2 = random.split(layer_rng)
w_1, b_1 = glorot_normal()(k1, (D + 1, hidden_dim)), normal()(k2, (hidden_dim,))

rng, layer_rng = random.split(rng)
k1, k2 = random.split(layer_rng)
w_2, b_2 = glorot_normal()(k1, (hidden_dim, D)), normal()(k2, (D,))

init_params = [(w_1, b_1), (w_2, b_2)]

# define ravel objects, booooooo
flat_params, ravel_params = ravel_pytree(init_params)
flat_y, ravel_y = ravel_pytree(pack_batch(rng)[1][0])

# optimizer
opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-2, gamma=0.99)
opt_state = opt_init(flat_params)

# ODE Integrator
nodeint = build_odeint(dynamics)
grad_nodes_loss = jax.jit(grad(nodes_predict_and_loss,argnums=-1))

assert parse_args.data_size % parse_args.batch_size == 0
batch_per_epoch = parse_args.data_size // parse_args.batch_size

itr = 0
for epoch in range(parse_args.nepochs):
  key = rng
  for batch in range(batch_per_epoch):
    itr+=1
    print("Iteration:",itr)
    key, batch_y = pack_batch(key)
    b_y0, b_y1 = batch_y

    # y_sol = nodeint(np.ravel(b_y0),ts,*flat_params) #(T,BS*D)
    # y_sol = np.array(jax.map(ravel_y,y_sol)) # (T,BS,D)
    # b_y1_pred = y_sol[-1]
    # loss_fun(b_y1_pred,b_y1)

    flat_params = get_params(opt_state)
    dldp = grad_nodes_loss(b_y1,np.ravel(b_y0), ts, flat_params)
    opt_state = opt_update(itr,dldp,opt_state)

    # print loss:
    if True:
      y_sol = nodeint(np.ravel(b_y0),ts,*flat_params) #(T,BS*D)
      y_sol = np.array(jax.map(ravel_y,y_sol)) # (T,BS,D)
      b_y1_pred = y_sol[-1]
      print(loss_fun(b_y1_pred,b_y1))


# One Trajectory
import matplotlib.pyplot as plt

plot_y0 =  np.expand_dims(np.repeat(d0,200),1)
plot_ts = np.linspace(0.,1.,100)
plot_sol = nodeint(np.ravel(plot_y0),plot_ts, *flat_params)

ys= plot_sol[:,1]

plt.clf()
plt.plot(plot_ts,ys)
plt.show()

ti = 50
d0 = ys[ti:ti+1]
t0 = plot_ts[ti]
(p1,[p2,p3,p4,p5,p6]) = plot_jet = sol_recursive(jet_wrap_predict(ravel_params(flat_params)),np.expand_dims(np.repeat(d0,200),1),t0)

d1 = p1[1,1]
d2 = p2[1,1]
d3 = p3[1,1]
d4 = p4[1,1]
d5 = p5[1,1]
d6 = p6[1,1]

plt.scatter(t0,d0)

tay_1 = lambda t : d0 + (t-t0) * d1
plt.plot(plot_ts,tay_1(plot_ts))

tay_2 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)
plt.plot(plot_ts,tay_2(plot_ts))

tay_3 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3)
plt.plot(plot_ts,tay_3(plot_ts))

tay_4 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3) + (t-t0)**4 * d4/fact(4)
plt.plot(plot_ts,tay_4(plot_ts))

tay_5 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3) + (t-t0)**4 * d4/fact(4)+ (t-t0)**5 * d5/fact(5)
plt.plot(plot_ts,tay_5(plot_ts))

tay_6 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3) + (t-t0)**4 * d4/fact(4)+ (t-t0)**5 * d5/fact(5) + (t-t0)**6 * d6/fact(6)
plt.plot(plot_ts,tay_6(plot_ts))


# Animation
anim_dir = 'anim'

y0 = 2.5
plot_y0 =  np.expand_dims(np.repeat(y0,200),1)
plot_ts = np.linspace(0.,1.,100)
plot_sol = nodeint(np.ravel(plot_y0),plot_ts, *flat_params)

ys= plot_sol[:,1]


compute_jet = lambda z,t : sol_recursive(jet_wrap_predict(ravel_params(flat_params)),np.expand_dims(np.repeat(z,200),1),t)
jit_jet = jax.jit(compute_jet)


# animation code
def plot_tays(ti):
  d0 = ys[ti:ti+1]
  t0 = plot_ts[ti]
  (p1,[p2,p3,p4,p5,p6]) =  jit_jet(d0,t0)

  d1 = p1[1,1]
  d2 = p2[1,1]
  d3 = p3[1,1]
  d4 = p4[1,1]
  d5 = p5[1,1]
  d6 = p6[1,1]


  def common_options(order):
    plt.xlim(-0.05,1.05)
    plt.ylim(-10,20)  
    plt.title(f"{order}-order Taylor Approximation to Neural ODE Solution")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend(loc=2)

  tay_1 = lambda t : d0 + (t-t0) * d1
  plt.clf()
  plt.plot(plot_ts,ys, label="Neural ODE Solution")
  plt.plot(plot_ts,tay_1(plot_ts), label="Taylor Approximation")
  plt.scatter(t0,d0, zorder=10)
  common_options(1)
  plt.savefig(os.path.join(anim_dir,'1',str(ti).zfill(4)))
  plt.close()

  tay_2 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)
  plt.clf()
  plt.plot(plot_ts,ys, label="Neural ODE Solution")
  plt.plot(plot_ts,tay_2(plot_ts), label="Taylor Approximation")
  plt.scatter(t0,d0, zorder=10)
  common_options(2)
  plt.savefig(os.path.join(anim_dir,'2',str(ti).zfill(4)))
  plt.close()

  tay_3 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3)
  plt.clf()
  plt.plot(plot_ts,ys, label="Neural ODE Solution")
  plt.plot(plot_ts,tay_3(plot_ts), label="Taylor Approximation")
  plt.scatter(t0,d0, zorder=10)
  common_options(3)
  plt.savefig(os.path.join(anim_dir,'3',str(ti).zfill(4)))
  plt.close()

  tay_4 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3) + (t-t0)**4 * d4/fact(4)
  plt.clf()
  plt.plot(plot_ts,ys, label="Neural ODE Solution")
  plt.plot(plot_ts,tay_4(plot_ts), label="Taylor Approximation")
  plt.scatter(t0,d0, zorder=10)
  common_options(4)
  plt.savefig(os.path.join(anim_dir,'4',str(ti).zfill(4)))
  plt.close()

  tay_5 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3) + (t-t0)**4 * d4/fact(4)+ (t-t0)**5 * d5/fact(5)
  plt.clf()
  plt.plot(plot_ts,ys, label="Neural ODE Solution")
  plt.plot(plot_ts,tay_5(plot_ts), label="Taylor Approximation")
  plt.scatter(t0,d0, zorder=10)
  common_options(5)
  plt.savefig(os.path.join(anim_dir,'5',str(ti).zfill(4)))
  plt.close()

  tay_6 = lambda t : d0 + (t-t0) * d1 + (t-t0)**2 * d2/fact(2)+ (t-t0)**3 * d3/fact(3) + (t-t0)**4 * d4/fact(4)+ (t-t0)**5 * d5/fact(5) + (t-t0)**6 * d6/fact(6)
  plt.clf()
  plt.plot(plot_ts,ys, label="Neural ODE Solution")
  plt.plot(plot_ts,tay_6(plot_ts), label="Taylor Approx.")
  plt.scatter(t0,d0, zorder=10)
  common_options(6)
  plt.savefig(os.path.join(anim_dir,'6',str(ti).zfill(4)))
  plt.close()

for ti in range(len(plot_ts)):
  plot_tays(ti)

### Integrate regularizer
def append_aug(y,r):
  """
  y::(BS,D)
  r::(BS,R)
  yr::(BS,D+R)
  """
  yr = np.concatenate((y,r),axis=1)
  return yr

def unpack_aug(yr):
  return yr[:,:D],yr[:,D:]

@jax.jit
def r2(y,t,params):
  return np.sum(sol_recursive(jet_wrap_predict(ravel_params(params)),y,t)[1][0]**2,axis=1,keepdims=True)

@jax.jit
def aug_dynamics(yr,t,*args):
  """
  yr::unravel((BS,D+R))
  t::scalar
  args::flat_params
  return::unravel((BS,D+R))
  """
  y,r = ravel_yr(yr)[:,:D], ravel_yr(yr)[:,D:] # because jax needs flat yr
  yt = append_time(y,t)
  dydt = predict(ravel_params(np.array(args)),yt)
  drdt = r2(y,t,np.array(args))

  return np.ravel(append_aug(dydt,drdt)) # jax needs flat y

nodeint_aug = jax.jit(build_odeint(aug_dynamics))

def aug_init(y):
  return append_aug(y,np.zeros_like(y))

flat_yr, ravel_yr = ravel_pytree(aug_init(b_y0))

@jax.jit
def loss_aug(y0, ts, params):
  # Integrate Augmented Dynamics
  yr_sol = nodeint_aug(np.ravel(aug_init(y0)),ts,*params)
  yr_sol = np.array(jax.map(ravel_yr,yr_sol))

  # Get final state
  yr1 = yr_sol[-1]

  # Separate state and augmented state
  y1_pred, r_pred = unpack_aug(yr1)

  # compute mean
  return np.mean(r_pred)

loss_aug(b_y0,ts,flat_params) 

dr2dp_func=jax.jit(grad(loss_aug, argnums=-1))

dr2dp = dr2dp_func(b_y0,ts,flat_params)


### For testing gradients 

# numerical grad test
import numpy as onp
def nd(f, x, eps=1e-5):
  flat_x, unravel = ravel_pytree(x)
  dim = len(flat_x)
  g = onp.zeros_like(flat_x)
  for i in range(dim):
    d = onp.zeros_like(flat_x)
    d[i] = eps
    g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
  return g


# test grad ode jet sol nn
dr2dp_func=jax.jit(grad(loss_aug, argnums=-1))
dr2dp = dr2dp_func(b_y0,ts,flat_params)

# make closure for numerical version
dr2_p_closure = jax.jit(lambda params : loss_aug(b_y0,ts,params))
dr2dp_numerical = nd(dr2_p_closure, flat_params)
