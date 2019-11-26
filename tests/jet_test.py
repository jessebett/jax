import numpy.random as npr
from scipy.special import factorial as fact

import jax
import jax.nn
from jax.util import safe_map
from jax import vjp,jvp, jet, grad
import jax.numpy as np

from jax.config import config
config.update("jax_enable_x64",True)

map = safe_map

def repeated(f, n):
  def rfun(p):
    return jax.reduce(lambda x, _: f(x), np.xrange(n), p)

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
  y, terms = jet(f, primals, series)
  y_jvp, terms_jvp = jvp_taylor(f, primals, series)
  # import ipdb; ipdb.set_trace()
  assert np.allclose(y, y_jvp)
  assert np.allclose(terms, terms_jvp, atol=atol)

def test_reshape():
  npr.seed(0)
  D = 10
  N = 6
  x = np.arange(D)
  terms_in = list(npr.randn(N,D))
  f = lambda x: np.reshape(x,(2,5))
  jvp_test_jet(f, (x, ), (terms_in, ), atol=1e-4)

def test_exp():
  npr.seed(0)
  D = 3
  N = 6
  x = npr.randn(D)
  terms_in = list(npr.randn(N,D))
  # x = 1.
  # terms_in = [2., 3., 4.]
  # x = np.array([1.,5.])
  # terms_in =[np.array([2.,6.]),np.array([3.,7.]),np.array([4.,8])]
  jvp_test_jet(np.exp, (x, ), (terms_in, ), atol=1e-4)

def test_mlp():
  sigm = lambda x: 1. / (1. + np.exp(-x))
  def mlp(M1,M2,x):
    return np.dot(sigm(np.dot(x,M1)),M2)
  f_mlp = lambda x: mlp(M1,M2,x)
  M1,M2 = (npr.randn(10,10), npr.randn(10,5))
  x= npr.randn(2,10)
  terms_in = [np.ones_like(x),np.zeros_like(x), np.zeros_like(x),np.zeros_like(x)]
  jvp_test_jet(f_mlp,(x,),[terms_in])

# def test_mlp_t():
#   def mlp(M1,M2,x,t):
#     x = np.append(x,t)
#     return M2*np.exp(M1*x)
#   f_mlp = lambda t: mlp(M1,M2,x,t)
#   M1,M2 = (npr.randn(10,11), npr.randn(10,10))
#   x= npr.randn(10)
#   t = npr.randn()
#   primals = (t,)
#   t_series = [np.ones_like(t)] + [np.zeros_like(t)]*3
#   series= [t_series]
#   y,terms = jet(f_mlp,primals,series)

def test_log():
  raise NotImplementedError


def test_tanh():
  raise NotImplementedError


def test_sin():
  N = 4
  # x = npr.randn()
  # terms_in = list(npr.randn(N))
  x = 2.
  terms_in = (1., 0., 0.)
  jvp_test_jet(np.sin, (x, ), (terms_in, ))


def test_cos():
  N = 4
  x = npr.randn()
  terms_in = list(npr.randn(N))
  jvp_test_jet(np.cos, (x, ), (terms_in, ), atol=1e-1)

def test_sqrt():
  N = 5
  x = npr.randn()**2
  terms_in = list(npr.randn(N))
  # print terms_in
  # print jet(np.sqrt, (x, ), (terms_in, ))
  # print jvp_taylor(np.sqrt, (x, ), (terms_in, ))
  jvp_test_jet(np.sqrt, (x, ), (terms_in, ), atol=1e-1)

# def test_exp():
#   N = 5
#   x = npr.randn()**2
#   terms_in = list(npr.randn(N))
#   jvp_test_jet(np.exp, (x, ), (terms_in, ), atol=1e-1)

def test_neg():
  N = 4
  x = npr.randn()
  terms_in = list(npr.randn(N))
  f = lambda x: -x
  jvp_test_jet(f, (x, ), (terms_in,), atol=1e-2)

def test_sum():
  D = 3
  N = 4
  x1 = npr.randn(D)
  x2 = npr.randn(D)
  f = lambda a, b: a + b
  primals = (x1, x2)
  terms_in1 = list(npr.randn(N,D))
  terms_in2 = list(npr.randn(N,D))
  series_in = (terms_in1, terms_in2)
  jvp_test_jet(f, primals, series_in)

def test_dot():
  D = 6
  N = 4
  x1 = npr.randn(D)
  x2 = npr.randn(D)
  primals = (x1, x2)
  terms_in1 = list(npr.randn(N,D))
  terms_in2 = list(npr.randn(N,D))
  series_in = (terms_in1, terms_in2)
  jvp_test_jet(np.dot, primals, series_in)


def test_mul():
  D = 3
  N = 4
  x1 = npr.randn(D)
  x2 = npr.randn(D)
  f = lambda a, b: a * b
  primals = (x1, x2)
  terms_in1 = list(npr.randn(N,D))
  terms_in2 = list(npr.randn(N,D))
  series_in = (terms_in1, terms_in2)
  jvp_test_jet(f, primals, series_in)

def test_div():
  D = 3
  N = 4
  # x1 = npr.randn(D)
  # x2 = npr.randn(D)
  x = 1.
  y = 5.
  primals = (x,y)
  # terms_in1 = list(npr.randn(N,D))
  # terms_in2 = list(npr.randn(N,D))
  terms_x = [1.,2.,3.,4.]
  terms_y = [6.,7.,8.,9.]
  series_in = (terms_x, terms_y)
  f = lambda a, b: a / b
  jvp_test_jet(f, primals, series_in)

def test_sigmoid():
  D1,D2 = 10,3
  N = 4
  x = npr.randn(D1,D2)
  # f = jax.nn.sigmoid #TODO: jax special expit
  f = lambda x: 1. / (1. + np.exp(-x))
  primals = (x,)
  terms_in = [npr.randn(D1,D2)]*N
  series_in = (terms_in,)
  jvp_test_jet(f, primals, series_in)

# TODO: matmul correctly handle other dims, or is this all dot?
# def test_mul2():
#   d1 = 2
#   d2 = 10
#   n = 4
#   x1 = npr.randn(d1,d2)
#   x2 = npr.randn(d2,d2)
#   f = lambda a, b: a * b
#   primals = (x1, x2)
#   terms_in1 = list(npr.randn(n,d1,d2))
#   terms_in2 = list(npr.randn(n,d2,d2))
#   series_in = (terms_in1, terms_in2)
#   jvp_test_jet(f, primals, series_in)

## Test Combinations?
def test_sin_sin():
  N = 4
  x = npr.randn()
  terms_in = npr.randn(N)
  f = lambda x: np.sin(np.sin(x))
  jvp_test_jet(f, (x, ), (terms_in, ), atol=1e-2)


def test_expansion():
  N = 4
  x = npr.randn()
  terms_in = [1., 0., 0., 0.]
  f = np.sin
  # print jet(f, (x, ), (terms_in, ))
  return expand(f, (x, ), (terms_in, ))
  # return jet(f,(x,),(terms_in,))


def expand(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  return grad(grad(grad(expansion)))(0.)


def test_sol_recursive():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    return z * z * 0.1

  # Initial  Conditions
  t0, z0 = 2., 1.

  # Closed-form solution
  def true_sol(t):
    return -(10. * z0) / (-10. + t * z0 - t0 * z0)

  # Evaluate at t_eval
  t_eval = 5.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, z0, [t0, t_eval])[1]
  assert np.isclose(z_eval_true, z_eval_odeint)

  # True derivatives of solution
  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0.,0.,0.), ))

  # Explicitly call coefficient functions
  def y0_coeff(x0):
    return jet(g, (x0, ), ((np.zeros_like(x0), ), ))[0]

  def y1_coeff(x0,x1):
    return jet(g, (x0, ), ((x1, ), ))[1]

  def y1_coeff(x0,x1):
    return jet(g, (x0, ), ((x1, ), ))[1]

  # First coeffs computed recursively
  (y0, [y1h]) = jet(g, (z_eval_true, ), ((1., ), ))
  (y0, [y1, y2h]) = jet(g, (z_eval_true, ), ((
      y0,
      y1h,
  ), ))
  (y0, [y1, y2, y3h]) = jet(g, (z_eval_true, ), ((y0, y1, y2h), ))
  (y0, [y1, y2, y3, y4h]) = jet(g, (z_eval_true, ), ((y0, y1, y2, y3h), ))
  (y0, [y1, y2, y3, y4, y5h]) = jet(g, (z_eval_true, ),
                                    ((y0, y1, y2, y3, y4h), ))
  print((y0, [y1, y2, y3, y4, y5h]),"\n")
  print(true_ds,"\n")


def test_sol_newtondouble_explicit():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    return z * z * 0.1

  # Initial  Conditions
  t0, z0 = 2., 1.

  # Closed-form solution
  def true_sol(t):
    return -(10. * z0) / (-10. + t * z0 - t0 * z0)

  # Evaluate at t_eval
  t_eval = 5.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, z0, [t0, t_eval])[1]
  assert np.isclose(z_eval_true, z_eval_odeint)

  # True derivatives of solution
  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0.,0.,0.), ))

  # By Newton doubling
  x0 = z_eval_true

  # first pass
  s = 0
  j = 2**(s + 1) - 1
  (y0h, [y1h]) = jet(g, (x0, ), ((0., ), ))
  # A0 = lambda v: jvp(y0_coeff,(x0,),(v,))[1]
  A0 = lambda v: jvp(lambda x0 : jet(g,(x0,),((0.,),)),(x0,),((v,)))[1][0]

  x1 = 1./1 * (y0h)
  x2 = 1./2 * (y1h + A0(x1))

  #second pass
  s=1
  j = 2**(s + 1) - 1
  (y0h, [y1h, y2h, y3h, y4h, y5h,y6h]) = jet(g,(x0,),((x1,x2 * fact(2),0.,0.,0.,0.),))
  #dy0/dx0*v
  A0 = lambda v: jvp(lambda x0 : jet(g,(x0,),((0.,),)),(x0,),((v,)))[1][0]
  #dy1/dx0*v
  A1 = lambda v: jvp(lambda x0 : jet(g,(x0,),((x1,),)),(x0,),((v,)))[1][1][0]
  #dy2/dx0*v
  A2 = lambda v: jvp(lambda x0 : jet(g,(x0,),((x1,x2),)),(x0,),((v,)))[1][1][1]

  x3 = 1./3 * (y2h/fact(2))
  x4 = 1./4 * (y3h/fact(3) + A0(x3))
  x5 = 1./5 * (y4h/fact(4) + A1(x3) + A0(x4))
  x6 = 1./6 * (y5h/fact(5) + A2(x3) + A1(x4) + A0(x5))

  # print x1,x2,x3,x4,x5,x6
  # print true_ds[1]

def test_sol_newtondouble_onevjp():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    return z * z * 0.1

  # Initial  Conditions
  t0, z0 = 2., 1.

  # Closed-form solution
  def true_sol(t):
    return -(10. * z0) / (-10. + t * z0 - t0 * z0)

  # Evaluate at t_eval
  t_eval = 5.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, z0, [t0, t_eval])[1]
  assert np.isclose(z_eval_true, z_eval_odeint)

  # True derivatives of solution
  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0.,0.,0.), ))

  # By Newton doubling
  x0 = z_eval_true

  # first pass
  s = 0
  j = 2**(s + 1) - 1
  (y0h, [y1h]) = jet(g, (x0, ), ((0., ), ))
  # A0 = lambda v: jvp(y0_coeff,(x0,),(v,))[1]
  A0 = lambda v: jvp(lambda x0 : jet(g,(x0,),((0.,),)),(x0,),((v,)))[1][0]

  x1 = 1./1 * (y0h)
  x2 = 1./2 * (y1h + A0(x1))

  #second pass
  s=1
  j = 2**(s + 1) - 1
  (y0h, [y1h, y2h, y3h, y4h, y5h,y6h]) = jet(g,(x0,),((x1,x2 * fact(2),0.,0.,0.,0.),))
  #dy0/dx0*v
  A0 = lambda v: jvp(lambda x0 : jet(g,(x0,),((0.,),)),(x0,),((v,)))[1][0]
  #dy1/dx0*v
  A1 = lambda v: jvp(lambda x0 : jet(g,(x0,),((x1,),)),(x0,),((v,)))[1][1][0]
  #dy2/dx0*v
  A2 = lambda v: jvp(lambda x0 : jet(g,(x0,),((x1,x2),)),(x0,),((v,)))[1][1][1]

  x3 = 1./3 * (y2h/fact(2))
  x4 = 1./4 * (y3h/fact(3) + A0(x3))
  x5 = 1./5 * (y4h/fact(4) + A1(x3) + A0(x4))
  x6 = 1./6 * (y5h/fact(5) + A2(x3) + A1(x4) + A0(x5))

  ## Newton doubling with vjp of jet?
  x0 = z_eval_true

  # first pass
  s=0
  j=2**(s+1)-1

  f_jet = lambda x0: jet(g, (x0,), ((0.,),))
  yhs, f_vjp = vjp(f_jet, *(x0,))

  x1 = 1./1 * yhs[0]
  x2 = 1./2 * (yhs[1][0] + f_vjp((x1,[0.]))[0])

  # second pass
  s=1
  j=2**(s+1)-1
  
  yhs = jet(g, (x0,), ((x1,x2*fact(2),0.,0.,0.),))

  f_jet = lambda x0: jet(g, (x0,), ((x1,x2*fact(2)),))
  yhs_wasted, f_vjp = vjp(f_jet, *(x0,))

  x3 = 1./3 * (yhs[1][1]/fact(2))
  x4 = 1./4 * (yhs[1][2]/fact(3) + f_vjp((x3,[0.,0.]))[0])
  x5 = 1./5 * (yhs[1][3]/fact(4) + f_vjp((x4,[x3,0.]))[0])
  x6 = 1./6 * (yhs[1][4]/fact(5) + f_vjp((x5,[x4,x3/fact(2)]))[0])

  ## Wait does this work?
  x0 = z_eval_true

  # first pass
  s=0
  j=2**(s+1)-1

  f_jet = lambda x0: jet(g, (x0,), ((1.,),))
  yhs, f_vjp = vjp(f_jet, *(x0,))

  x1 = 1./1 * yhs[0]
  x2 = 1./2 * (yhs[1][0] + f_vjp((x1,[0.]))[0])

  # second pass
  s=1
  j=2**(s+1)-1
  
  yhs = jet(g, (x0,), ((x1,x2*fact(2),0.,0.,0.),))

  f_jet = lambda x0: jet(g, (x0,), ((x1,x2*fact(2)),))
  yhs_wasted, f_vjp = vjp(f_jet, *(x0,))

  x3 = 1./3 * (yhs[1][1]/fact(2))
  x4 = 1./4 * (yhs[1][2]/fact(3) + f_vjp((x3,[0.,0.]))[0])
  x5 = 1./5 * (yhs[1][3]/fact(4) + f_vjp((x4,[x3,0.]))[0])
  x6 = 1./6 * (yhs[1][4]/fact(5) + f_vjp((x5,[x4,x3/fact(2)]))[0])

def test_sol2():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f(z, t):
    return g(z)

  def g(z):
    z0, z1, t = z
    return np.array([np.sin(2.*t),np.cos(2.*t),1.])

  def g_sad(z):
    z0 = np.dot(np.array([1.,0.,0.]),z)
    z1 = np.dot(np.array([0.,1.,0.]),z)
    t = np.dot(np.array([0.,0.,1.]),z)
    return np.array([np.sin(2.*t),np.cos(2.*t),1.])

  # Initial  Conditions
  t0, z0 = 2., (0.,0.)

  # Closed-form solution
  def true_sol(t):
    return np.array([
        z0[0] + 0.5*np.cos(2.*t0) - 0.5*np.cos(2.*t)*np.cos(2.*t0) + 0.5*np.sin(2.*t) * np.sin(2.*t0),
        0.5*(2.*z0[1] + np.sqrt(1. - np.cos(2.*t0)**2) + np.cos(2.*t0)*np.sin(2.*t) + np.cos(2.*t)*np.sin(2.*t0)),
        t0 +t
        ])


  # Evaluate at t_eval
  t_eval = 3.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f, (z0[0],z0[1],t0), [t0, t_eval+ t0])[1]
  assert np.allclose(np.array(z_eval_true), z_eval_odeint)

  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0.,0.,0.), ))

  # First coeffs computed recursively
  x0 = np.array(z_eval_true)
  (y0, [y1h]) = jet(g, (x0,), ((np.ones_like(x0),),))

  ## Newton doubling with vjp of jet?
  x0 = np.array(z_eval_true)

  # first pass
  s=0
  j=2**(s+1)-1

  f_jet = lambda x0: jet(g, (x0,), ((np.zeros_like(x0),),))
  yhs, f_vjp = vjp(f_jet, *(x0,))
  
  x1 = 1./1 * yhs[0]
  x2 = 1./2 * (yhs[1][0] + f_vjp((x1, [np.zeros_like(x1)] * len(yhs[1])))[0])

  # second pass
  s=1
  j=2**(s+1)-1
  
  yhs = jet(g, (x0,), ([x1, x2*fact(2)] + [np.zeros_like(x1)] * 3,))

  f_jet = lambda x0: jet(g, (x0,), ((x1,x2*fact(2)),))
  yhs_wasted, f_vjp = vjp(f_jet, *(x0,))

  x3 = 1./3 * (yhs[1][1]/fact(2))
  x4 = 1./4 * (yhs[1][2]/fact(3) + f_vjp((x3,[np.zeros_like(x3)] * 2))[0])
  x5 = 1./5 * (yhs[1][3]/fact(4) + f_vjp((x4,[x3,np.zeros_like(x3)]))[0])
  x6 = 1./6 * (yhs[1][4]/fact(5) + f_vjp((x5,[x4,x3/fact(2)]))[0])



def test_sol2_tup():
  from scipy.integrate import odeint

  ## IVP with known solution
  def f_tup(z, t):
    return g_tup(*z)

  def g_tup(z0,z1,zt):
    return (np.sin(2.*zt), np.cos(2.*zt),1.)

  
  # Initial  Conditions
  t0, z0 = 2., (0.,0.)

  # Closed-form solution
  def true_sol(t):
    return np.array([ 
        z0[0] + 0.5*np.cos(2.*t0) - 0.5*np.cos(2.*t)*np.cos(2.*t0) + 0.5*np.sin(2.*t) * np.sin(2.*t0),
        0.5*(2.*z0[1] + np.sqrt(1. - np.cos(2.*t0)**2) + np.cos(2.*t0)*np.sin(2.*t) + np.cos(2.*t)*np.sin(2.*t0)),
        t0 +t
        ])


  # Evaluate at t_eval
  t_eval = 3.
  z_eval_true = true_sol(t_eval)

  # Use numerical integrator to test true solution
  z_eval_odeint = odeint(f_tup, (z0[0],z0[1],t0), [t0, t_eval+ t0])[1]
  assert np.allclose(np.array(z_eval_true), z_eval_odeint)

  true_ds = jvp_taylor(true_sol, (t_eval, ), ((1., 0., 0., 0., 0.,0.,0.), ))

  ## Newton doubling with vjp of jet?
  x0 = np.array(z_eval_true)

  # first pass
  s=0
  j=2**(s+1)-1

  f_jet = lambda x00,x01,x02: jet(g_tup, (x00,x01,x02), ((0.,),(0.,),(0.,),))
  yhs, f_vjp = vjp(f_jet, *x0)

  x1 = 1./1 * yhs[0]
  x2 = 1./2 * (yhs[1][0] + f_vjp((x1,[0.]))[0])

  # second pass
  s=1
  j=2**(s+1)-1
  
  yhs = jet(g, (x0,), ((x1,x2*fact(2),0.,0.,0.),))

  f_jet = lambda x0: jet(g, (x0,), ((x1,x2*fact(2)),))
  yhs_wasted, f_vjp = vjp(f_jet, *(x0,))

  x3 = 1./3 * (yhs[1][1]/fact(2))
  x4 = 1./4 * (yhs[1][2]/fact(3) + f_vjp((x3,[0.,0.]))[0])
  x5 = 1./5 * (yhs[1][3]/fact(4) + f_vjp((x4,[x3,0.]))[0])
  x6 = 1./6 * (yhs[1][4]/fact(5) + f_vjp((x5,[x4,x3/fact(2)]))[0])


def test_jvp_vjp_sum_relationship():
  def F(x,y):
    return np.sin(np.sin(x**2) * np.cos(y**3))

  primals = (2.,3.)
  tangents_in = (0.1,0.2)
  tangents_out = 0.5

  fwd = jvp(F,primals,tangents_in)[1] * tangents_out
  rev = np.sum(np.array(vjp(F,*primals)[1](tangents_out)) * np.array([tangents_in]))

  assert fwd == rev


# def test_vector_sin():
#   D = 10
#   x = npr.randn(D)

#   def f(x): return np.sin(x)

#   vs = [x, npr.randn(D), np.zeros(D), np.zeros(D), np.zeros(D)]
#   terms = prop(make_derivs_sin, vs)
#   assert np.isclose(terms, jvps(f, x, vs[1], 4)).all()

# def test_vector_sin_sin():
#   D = 1
#   x = npr.randn(D)

#   def f(x): return np.sin(np.sin(x))

#   vs = [x, np.ones(D), np.zeros(D), np.zeros(D), np.zeros(D)]
#   terms1 = prop(make_derivs_sin, vs)
#   terms2 = prop(make_derivs_sin, terms1)
#   assert np.isclose(terms2, jvps(f, x, vs[1], 4)).all()
