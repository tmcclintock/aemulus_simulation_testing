import corner
import numpy as np

def docorner(i):
    data = np.loadtxt("chain2_defg_box%d.txt"%i)
    fig = corner.corner(data)
    fig.savefig("corner_test_box%d.png"%i)

if __name__=="__main__":
    docorner(0)
