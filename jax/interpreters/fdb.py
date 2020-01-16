from functools import partial
from collections import Counter

import numpy as onp
from scipy.special import factorial as fact

from jax import core
# from ..core import pack
from jax.util import unzip2, prod
import jax.linear_util as lu


@lu.transformation
def jet(primals, series):
  with core.new_master(JetTrace) as master:
    trace = JetTrace(master, core.cur_sublevel())
    in_tracers = map(partial(JetTracer, trace), primals, series)
    ans = yield in_tracers, {}
    out_tracer = trace.full_raise(ans)
    out_primal, out_terms = out_tracer.primal, out_tracer.terms
  yield out_primal, out_terms


class JetTracer(core.Tracer):
  __slots__ = ["primal", "terms"]

  def __init__(self, trace, primal, terms):
    assert type(terms) in (ZeroSeries, list, tuple)
    self.trace = trace
    self.primal = primal
    self.terms = terms

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def unpack(self):
    terms_transposed = zip(*self.terms)
    return map(partial(JetTracer, self.trace), self.primal, terms_transposed)

  def full_lower(self):
    return self  # TODO symbolic zeros

class JetTrace(core.Trace):

  def pure(self, val):
    return JetTracer(self, val, zero_series)

  def lift(self, val):
    return JetTracer(self, val, zero_series)

  def sublift(self, val): return JetTracer(self, val.primal, val.terms)

  def process_primitive(self, primitive, tracers, params):
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    order, = {len(terms) for terms in series_in if terms is not zero_series}
    series_in = [[zero_term] * order if s is zero_series else s
                 for s in series_in]
    series_in = [[onp.zeros(onp.shape(x), dtype=onp.result_type(x)) if t is zero_term else t for t in series]
                 for x, series in zip(primals_in, series_in)]
    if primitive in prop_rules.keys():
      primal_out, terms_out = prop_rules[primitive](primals_in, zip(*series_in), **params)
      return JetTracer(self, primal_out, terms_out)
    #TODO: Unify previous fdb with fallback to generic prop
    else:
      primal_out, derivs = jet_rules[primitive](primals_in, order, **params)
      terms_out = prop(derivs, tuple(zip(*series_in)))
      return JetTracer(self, primal_out, terms_out)

  # def pack(self, tracers):
  #   primals = pack(t.primal for t in tracers)
  #   terms = pack(t.terms for t in tracers)
  #   import pdb; pdb.set_trace()
  #   return JetTracer(self, primals, terms)

  def process_call(self, call_primitive, f, tracers, params):
    assert False

  def post_process_call(self, call_primitive, out_tracer, params):
    assert False

  def join(self, xt, yt):
    assert False


class ZeroTerm(object): pass
zero_term = ZeroTerm()

class ZeroSeries(object): pass
zero_series = ZeroSeries()


jet_rules = {}
prop_rules = {}


### utilities

def product(xs):
  if any(x is zero_term for x in xs):
    return zero_term
  else:
    return prod(xs)

# from http://jeromekelleher.net/generating-integer-partitions.html
def accel_asc(n):
  a = [0 for i in range(n + 1)]
  k = 1
  y = n - 1
  while k != 0:
    x = a[k - 1] + 1
    k -= 1
    while 2 * x <= y:
      a[k] = x
      y -= x
      k += 1
    l = k + 1
    while x <= y:
      a[k] = x
      a[l] = y
      yield a[:k + 2]
      x += 1
      y -= 1
    a[k] = x + y
    y = x + y - 1
    yield a[:k + 1]

def partitions(k):
  if k == 0:
    return iter([[]])
  else:
    return accel_asc(k)

def sym(sigma):
  denom = prod(fact(count) for _, count in Counter(sigma).items())
  return fact(sum(sigma)) / prod(map(fact, sigma)) / denom

def tensor_coefficients(derivs,k):
  return lambda terms: sum(derivs[len(sigma)-1]([terms[i-1] for i in sigma]) * sym(sigma)
              for sigma in partitions(k))

def prop_new(derivs, terms):
  return [tensor_coefficients(derivs,k) for k in range(1, len(terms) + 1)]

def prop(derivs, terms):
  return [sum(derivs[len(sigma)-1]([terms[i-1] for i in sigma]) * sym(sigma)
              for sigma in partitions(k))
          for k in range(1, len(terms) + 1)]

def tay_to_deriv_coeff(u_tay):
  u_deriv = [ui*fact(i) for (i,ui) in enumerate(u_tay)]
  return u_deriv

def deriv_to_tay_coeff(u_deriv):
  u_tay = [ui/fact(i) for (i,ui) in enumerate(u_deriv)]
  return u_tay

def taylor_tilde(u_tay):
  u_tilde = [i * ui for (i,ui) in enumerate(u_tay)]
  return u_tilde

def taylor_untilde(u_tilde):
  u_tay = [i * ui for (i,ui) in enumerate(u_tilde)]
  return u_tay
