"""
Take some results, such as efg, and rotate the chains to break tight correlations.
"""
import numpy as np
import pandas as pd
import corner, sys, os
import matplotlib.pyplot as plt

inpath = "../chains/chain_np%d_mi%d_box%d.txt"
opath = "./rchain_np%d_mi%d_box%d" #save with np.save
Rpath = "./R_np%d_mi%d.txt" #rotation matrix

nw = 48 #number of walkers
nburn = 1000 #steps/walker to burn
N = 40 #number of sims

def make_rotation_matrix(npars, mi, i=34):
    data = pd.read_csv(inpath%(npars, mi, i), dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    data = data[nw*nburn:]
    D = np.copy(data)
    C = np.cov(D,rowvar=False)
    w,Ri = np.linalg.eig(C)
    np.savetxt(Rpath%(npars,mi),Ri)
    print "Saved R at :\n\t%s"%(Rpath%(npars,mi))

def rotate_all_chains(Npars, Mi):
    R = np.loadtxt(Rpath%(npars,mi))
    true_means = np.zeros((N, Npars))
    mean_models = np.zeros((N, Npars))
    var_models  = np.zeros((N, Npars))
    for i in range(N):
        data = pd.read_csv(inpath%(Npars,Mi,i), dtype='float64', delim_whitespace=True)
        data = data.as_matrix()
        data = data[nw*nburn:]
        rD = np.dot(data[:],R) #rotated
        true_means[i] = np.mean(data, 0)
        mean_models[i] = np.mean(rD, 0)
        var_models[i] = np.var(rD, 0)
        np.save(opath%(Npars,Mi,i), rD)
        print "Saved np%d mi%d box%d"%(Npars, Mi, i)
    np.savetxt("np%d_mi%d_means.txt"%(Npars,Mi), true_means)
    np.savetxt("r_np%d_mi%d_means.txt"%(Npars,Mi), mean_models)
    np.savetxt("r_np%d_mi%d_vars.txt"%(Npars,Mi), var_models)
    print "Means saved at:\n\t%s"%("r_np%d_mi%d_means.txt"%(Npars,Mi))

def make_corner(npars, mi, i):
    R = np.loadtxt(Rpath%(npars,mi))
    data = pd.read_csv(inpath%(npars,mi,i), dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    data = data[nw*nburn:]
    fig = corner.corner(data)
    plt.gcf().savefig("corner_norot_np%d_mi%d_box%d.png"%(npars,mi,i))
    plt.clf()
    rD = np.dot(data[:],R) #rotated
    fig = corner.corner(rD)
    plt.gcf().savefig("corner_rot_np%d_mi%d_box%d.png"%(npars,mi,i))
    plt.clf()

def make_real_corner(npars, mi, i):
    R = np.loadtxt(Rpath%(npars,mi))
    data = pd.read_csv(inpath%(npars,mi,i), dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    data = data[nw*nburn:]
    rD = np.dot(data[:],R) #rotated
    from chainconsumer import ChainConsumer
    labs=[r"$d_1'$",r"$e_0'$", r"$e_1'$",r"$f_0'$",r"$g_0'$",r"$g_1'$"]
    labs=[r"$e_0'$",r"$f_0'$",r"$g_0'$",r"$d_1'$", r"$e_1'$", r"$g_1'$"]
    c = ChainConsumer()
    c.add_chain(rD, parameters=labs)
    c.configure(kde=True, tick_font_size=10, label_font_size=24, max_ticks=3, sigmas=[0,1,2,3], usetex=True)#, statistics='max_symmetric')
    fig = c.plotter.plot()#legend=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig("fig_Rcorner.pdf", bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    npars = 6
    mi = 5
    #make_rotation_matrix(npars, mi, 34)
    #rotate_all_chains(npars, mi)
    #make_corner(npars, mi, 0)
    make_real_corner(npars, mi, 0)
