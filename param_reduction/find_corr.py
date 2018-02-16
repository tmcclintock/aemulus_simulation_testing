import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt

model = "defg"
nw = 48 #number of walkers
nburn = 1000*nw - 1 #pandas already drops the top row

def get_chain(box):
    inpath = "../chains/chain2_%s_box%d.txt"%(model, box)
    data = pd.read_csv(inpath, dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    return data[nburn:]

def show_corner(data):
    fig = corner.corner(data)
    plt.show()

def diag(box):
    data = get_chain(box)
    e0 = data[:,2]
    f0 = data[:,4]
    ar = np.array([e0,f0]).T
    print_corr(ar)
    show_corner(ar)
    C = np.cov(ar, rowvar=False)
    w, R = np.linalg.eig(C)
    rdata = np.dot(ar[:], R)
    print_corr(rdata)
    show_corner(rdata)

def print_corr(ar):
    C = np.cov(ar, rowvar=False)
    D = np.diag(np.sqrt(C.diagonal()))
    iD = np.linalg.inv(D)
    Corr = np.dot(iD, np.dot(C, iD))
    print Corr

if __name__ == "__main__":
    diag(0)
