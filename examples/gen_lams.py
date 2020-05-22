import numpy as np

REGS = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r45"]

pt = 1.66810054e+00
delta = 1

for reg in REGS:
    filename = "%s_lams.txt" % reg
    np.savetxt(filename, np.logspace(-9, 3, num=51))

filename = "lr.txt"
np.savetxt(filename, np.logspace(-3, 1, num=16))
