import numpy as np
import corner
import matplotlib.pyplot as plt
import sys

nburn = 500
nw = 48
def make_corner(path, opath):
    c = np.loadtxt(path)[nburn*nw:]
    f = corner.corner(c)
    f.savefig(opath)
    #plt.show()
    plt.clf()
    
if __name__ == "__main__":
    combos = [[8,0],
              [7,5],
              [7,6],
              [6,3],
              [6,4],
              [6,5],
              [6,10],
              [6,15]]
    #npars = int(sys.argv[1])
    #mi = int(sys.argv[2])
    box = 10#int(sys.argv[3])
    for npars, mi in combos:
        name = "np%d_mi%d_box%d"%(npars,mi,box)
        path = "chains/chain_%s.txt"%name
        opath = "figs/corner_%s.png"%name
        make_corner(path,opath)
