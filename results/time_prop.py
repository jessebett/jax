import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr

exp_times = []
for N in range(20):
  npr.seed(0)
  x = npr.randn()
  terms_in = list(npr.randn(N))
  time_i = %timeit -o jax.jet(np.exp,(x,),(terms_in,))
  exp_times.append(time_i.best)

# onp.save("results/exp_prop",np.array(exp_times))
onp.save("results/exp_fdb",np.array(exp_times))

exp_prop = np.load("results/exp_prop.npy")
exp_fdb = np.load("results/exp_fdb.npy")

import matplotlib.pyplot as plt

plt.clf()
plt.semilogy(exp_prop, label="custom prop")
plt.semilogy(exp_fdb, label="generic fdb")
plt.legend()
plt.xticks(range(0,20))
plt.xlabel("order of differentation")
plt.ylabel("time in ms")
plt.title("exp primitive")
plt.savefig("results/exp_compare.pdf")
