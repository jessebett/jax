from functools import partial
from jax import vmap
from lax import *
import jax.numpy as np
from ..interpreters import fdb
from ..interpreters import xla
from custom_props import prop_exp
from copy import copy
from scipy.special import factorial as fact


def deflinear(prim):
  fdb.jet_rules[prim] = partial(linear_jet, prim)

def linear_jet(prim, primals, order, **params):
  ans = prim.bind(*primals, **params)
  def fst(vs):
    vs, = vs
    return prim.bind(*vs, **params)
  def nth(vs):
    return np.zeros_like(ans)
  derivs = itertools.chain([fst], itertools.repeat(nth))
  return ans, list(itertools.islice(derivs, order))

deflinear(neg_p)
deflinear(slice_p)
deflinear(xla.device_put_p)
deflinear(reshape_p)
deflinear(concatenate_p)  # TODO


def make_derivs_sin(primals, order):
  x, = primals
  sin_x = sin(x)
  def derivs():
    cos_x = cos(x)
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * cos_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -sin_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -cos_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * sin_x
  derivs = list(itertools.islice(itertools.cycle(derivs()), order))
  return sin_x, derivs
fdb.jet_rules[sin_p] = make_derivs_sin

def make_derivs_cos(primals, order):
  x, = primals
  cos_x = cos(x)
  def derivs():
    sin_x = sin(x)
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -sin_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -cos_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * sin_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * cos_x
  derivs = list(itertools.islice(itertools.cycle(derivs()), order))
  return cos_x, derivs
fdb.jet_rules[cos_p] = make_derivs_cos

def make_derivs_sqrt(primals,order,**params):
  x, = primals
  out = np.sqrt(x)
  #TODO: make generator dependent on order...
  def fst(vs):
    return fdb.product(map(operator.itemgetter(0), vs)) * 1./(2*x**(1./2))
  def snd(vs):
    return fdb.product(map(operator.itemgetter(0), vs)) * - 1./(4*x**(3./2))
  def thd(vs):
    return fdb.product(map(operator.itemgetter(0), vs)) * 3./(8*x**(5./2))
  def fth(vs):
    return fdb.product(map(operator.itemgetter(0), vs)) * - 15./(16*x**(7./2))
  def fith(vs):
    return fdb.product(map(operator.itemgetter(0), vs)) * 105./(32*x**(9./2))

  return out, [fst,snd,thd,fth,fith]
fdb.jet_rules[sqrt_p] = make_derivs_sqrt

# Generic Prop, deprecated
# def make_derivs_exp(primals,order,**params):
#   x, = primals
#   out = np.exp(x)
#   #TODO: make generator dependent on order...
#   def nth(vs):
#     return fdb.product(map(operator.itemgetter(0), vs)) * out
#   derivs = itertools.chain(itertools.repeat(nth))
#   return out, list(itertools.islice(derivs,order))
# fdb.jet_rules[exp_p] = make_derivs_exp
def prop_exp(primals_in, series_in):
  #TODO: refactor so no distinction between primals and terms
  #TODO: use numpy convolve
  #TODO: use scipy fft?
  x, = primals_in
  u = [primals_in] + series_in
  
  v = copy(u) 
  primals_out = np.exp(x)
  v[0] = primals_out
  for k in range(1,len(v)):
    def scale(j):
      return 1./(fact(k-j)*fact(j-1))
    v[k] = fact(k-1)*sum([scale(j)* v[k-j]*u[j][0] for j in range(1,k+1)])

  import ipdb; ipdb.set_trace()
  terms_out = v[1:]
    
  return primals_out, terms_out
fdb.prop_rules[exp_p] = prop_exp

def make_derivs_mul(primals, order):
  a, b = primals
  def fst(vs):
    (va, vb), = vs
    return va * b + a * vb
  def snd(vs):
    (v0a, v0b), (v1a, v1b) = vs
    return v0a * v1b + v1a * v0b
  def nth(vs):
    return np.zeros_like(a)
  derivs = itertools.chain([fst,snd], itertools.repeat(nth))
  return mul(a, b), list(itertools.islice(derivs, order))
fdb.jet_rules[mul_p] = make_derivs_mul

def manual_mul_conv(u,w):
  v = copy(u)
  for k in range(0,len(v)):
    def scale(j):
      # return 1.
      return 1./(fact(k-j)*fact(j))
    # v[k] = sum([scale(j)* u[j] * w[k-j] for j in range(0,k+1)])
    v[k] = fact(k)*sum([scale(j)* u[j] * w[k-j] for j in range(0,k+1)])
  return v

def mul_conv(u,w):
  # prepare for expected convolution dimensions
  u = np.array(u).T
  u = u[:,np.newaxis,:]
  w = np.array(w).T
  w = w[:,np.newaxis,:]
  w = np.flip(w,2)
  def batchwise_conv(ui,wi):
    # seems silly to produce new axes here
    ui = ui[np.newaxis,:]
    wi = wi[np.newaxis,:]
    return conv_general_dilated(ui,wi,(1,),[(4,0)])
  v = vmap(batchwise_conv)(u,w)
  # remove extra dimensions
  # is this robust?
  v = v[:,0,0,:]
  # Do we really want list?
  return list(v.T)

def prop_mul(primals_in, series_in):
  #TODO: refactor so no distinction between primals and terms
  u0, w0 = primals_in
  vu, vw = zip(*series_in)
  u = [u0,] + list(vu)
  w = [w0,] + list(vw)
  import ipdb; ipdb.set_trace()
  v = manual_mul_conv(u,w)
  # v_conv = mul_conv(u,w)
  print v
  return v[0], v[1:]
fdb.prop_rules[mul_p] = prop_mul

def make_derivs_dot(primals, order, **params):
  a, b = primals
  out = dot(a, b)
  def fst(vs):
    (va, vb), = vs
    return dot(va, b) + dot(a, vb)
  def snd(vs):
    (v0a, v0b), (v1a, v1b) = vs
    return dot(v0a, v1b) + dot(v1a, v0b)
  def nth(vs):
    return np.zeros_like(out)
  derivs = itertools.chain([fst, snd], itertools.repeat(nth))
  return out, list(itertools.islice(derivs, order))
fdb.jet_rules[dot_p] = make_derivs_dot



# from scipy.integrate import odeint as odeint_impl
#
# def odeint(f,z0,t, rtol=1e-7,atol=1e-9):
#   return odeint_p.bind(f,z0,t,rtol = rtol,atol=atol)
# odeint_p = Primitive('odeint')
# odeint_p.def_impl(odeint_impl)
# def make_derivs_odeint(primals,order):
#   def nth(vs):
#     return onp.zeros_like(a)
#   derivs = itertools.chain(itertools.repeat(nth))
#   return primals, list(itertools.islice(derivs, order))
# fdb.jet_rules[odeint_p] = make_derivs_odeint

def sol(f,z,t):
  sol_p.bind(f,z,t)

def sol_impl(f,z,t):
  return f(z,t)

sol_p = Primitive('sol')
sol_p.def_impl(sol_impl)
def make_derivs_sol(primals,order):
  f,z,t = primals
  def nth(vs):
    return np.zeros_like(z)
  derivs = itertools.chain(itertools.repeat(nth))
  import pdb; pdb.set_trace()
  return f(z,t), list(itertools.islice(derivs, order))
fdb.jet_rules[sol_p] = make_derivs_sol

