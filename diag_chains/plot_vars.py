import numpy as np
import matplotlib.pyplot as plt

name = "defg"

means = np.loadtxt("r_%s_means.txt"%name)
var = np.loadtxt("r_%s_vars.txt"%name)
std = np.sqrt(var)

n = np.arange(len(means))
print means.shape, std.shape, var.shape, n.shape
for i in range(len(means[0])):
    plt.errorbar(n, means[:,i], std[:,i])
plt.show()
