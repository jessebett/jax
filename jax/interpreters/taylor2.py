from functools import partial

from jax import core
from jax import linear_util as lu



# # takes nested list, returns nested list
# def foo(nested_list):
#   # ...

# def foo_vector(vec):
#   nested_list = unflatten(vec, INPUT_STRUCTURE)
#   out_nested_list = foo(nested_list)
#   out_vec, OUTPUT_STRUCTURE = flatten(out_nested_list)
#   return out_vec


@lu.transformation
def taylor2(primal, term1, term2):
  with core.new_master(Taylor2Trace) as master:
    trace = Taylor2Trace(master, core.cur_sublevel())
    in_tracer = Taylor2Tracer(trace, primal, (term1, term2))
    ans = yield (in_tracer,), {}
    out_tracer = trace.full_raise(ans)
    out_primal, out_terms = out_tracer.primal, out_tracer.terms
  yield out_primal, out_terms


class Taylor2Tracer(core.Tracer):
  __slots__ = ["primal", "terms"]

  # If this tracer corresponds to a value in the program computed as y = f(x),
  # then `primal` will have the value y, and `terms` will be a two-element tuple
  # with components `(\partial f(x) v,  v' partial^2 f(x) v)`.

  def __init__(self, trace, primal, terms):
    self.trace = trace
    self.primal = primal
    self.terms = terms

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def unpack(self):
    # x is a Taylor2Tracer tracing a tuple value
    # a, b, c = x
    #
    # return [Taylor2Tracer(self.trace, self.primals[i], (self.terms[0][i], self.terms[1][i]))
    #         for i in range(len(self.primal))]
    primals = tuple(self.primal)
    all_terms = zip(*self.terms)
    return map(partial(Taylor2Tracer, self.trace), primals, all_terms)

  def full_lower(self):
    if self.terms is symbolic_zero:
      return core.full_lower(self.primal)
    else:
      return self


class Taylor2Trace(core.Trace):

  def pure(self, val):
    return Taylor2Tracer(self, val, symbolic_zero)

  def lift(self, val):
    return Taylor2Tracer(self, val, symbolic_zero)

  def sublift(self, val):
    assert Taylor2Tracer(self, val.val, val.terms)

  def process_primitive(self, primitive, tracers, params):
    primals_in = [t.primal for t in tracers]
    series_in = [t.terms for t in tracers]
    taylor_rule = taylor_rules[primitive]
    primal_out, terms_out = taylor_rule(primals_in, series_in, **params)
    return Taylor2Tracer(self, primal_out, terms_out)

  def pack(self, tracers):
    assert False

  def process_call(self, call_primitive, f, tracers, params):
    assert False

  def post_process_call(self, call_primitive, out_tracer, params):
    assert False

  def join(self, xt, yt):
    assert False


class SymbolicZero(object): pass
symbolic_zero = SymbolicZero()


taylor_rules = {}
