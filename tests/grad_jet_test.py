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

def test_grad_mlp():
  sigm = lambda x: 1. / (1. + np.exp(-x))
  def mlp(M1,M2,x):
    return np.sum(np.dot(sigm(np.dot(x,M1)),M2))
  f_mlp = lambda x: mlp(M1,M2,x)
  M1,M2 = (npr.randn(10,10), npr.randn(10,5))
  x= npr.randn(2,10)
  terms_in = [np.ones_like(x)]+ [np.zeros_like(x)]*5
  # jvp_test_jet(f_mlp,(x,),[terms_in])

  def jvp_mlp(M1):
    return jvp_taylor(lambda x: mlp(M1,M2,x), (x,), (terms_in,))

  def jet_mlp(M1):
    return jet(lambda x: mlp(M1,M2,x), (x,), (terms_in,))


  # grad through primal of jvp is the same as grad through primal
  assert np.allclose(grad(lambda m1 : jvp_mlp(m1)[0])(M1),grad(lambda m1 : mlp(m1,M2,x))(M1))

  # grad through primal of jet is the same as grad through primal
  # BROKE! :(
  assert np.allclose(grad(lambda m1 : jet_mlp(m1)[0])(M1),grad(lambda m1 : mlp(m1,M2,x))(M1))

  # Jet and JVP for mlp are same
  assert np.allclose(jet_mlp(M1)[1],jvp_mlp(M1)[1])

  # Meat and Potatoes of it all:
  # gradient of second x-derivative wrt m1 with jvp
  d2fdm_jvp = grad(lambda m1 : jvp_mlp(m1)[1][0])(M1)
  # gradient of third x-derivative wrt m1 with jvp
  d3fdm_jvp = grad(lambda m1 : jvp_mlp(m1)[1][1])(M1)
  # gradient of fifth x-derivative wrt m1 with jvp
  d5fdm_jvp = grad(lambda m1 : jvp_mlp(m1)[1][4])(M1)

  # gradient of second x-derivative wrt m1 with jet
  d2fdm_jet = grad(lambda m1 : jet_mlp(m1)[1][0])(M1)
  # gradient of third x-derivative wrt m1 with jet
  d3fdm_jet = grad(lambda m1 : jet_mlp(m1)[1][1])(M1)
  # gradient of fifth x-derivative wrt m1 with jet
  d5fdm_jet = grad(lambda m1 : jet_mlp(m1)[1][4])(M1)

  assert np.allclose(d2fdm_jet,d2fdm_jvp)
  assert np.allclose(d3fdm_jet,d3fdm_jvp)
  assert np.allclose(d5fdm_jet,d5fdm_jvp)

