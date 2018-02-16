import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name = "dfg"
chainpath = "chain2_"+name+"_box%d.txt"

nw = 48 #nwalkers
nburn = nw*2000 -1

def make_means(N, Np):
    med_models = np.zeros((N, Np))
    mean_models = np.zeros((N, Np))
    var_models = np.zeros_like(mean_models)
    for i in range(0, N):
        print "Loading box%d"%i
        data = np.loadtxt(chainpath%i)[1000*nw:] #Remove burn-in
        print "shape:",data.shape
        mean_models[i] = np.mean(data, 0)
        med_models[i] = np.median(data, 0)
        var_models[i] = np.var(data, 0)
    np.savetxt("%s_medians.txt"%name, med_models)
    np.savetxt("%s_means.txt"%name, mean_models)
    np.savetxt("%s_vars.txt"%name, var_models)

def print_means():
    m = np.loadtxt("%s_means.txt"%name).T
    v = np.loadtxt("%s_vars.txt"%name).T
    ls = ['d0','d1','e0','e1','f0','f1','g0','g1']
    for i in range(len(ls)):
        print "%s %.4f %.2f %.2f"%(ls[i], np.mean(m[i]), np.std(m[i]), np.sqrt(np.mean(v[i])))
    return

def plot_means():
    me = np.loadtxt("%s_medians.txt"%name).T
    m = np.loadtxt("%s_means.txt"%name).T
    v = np.loadtxt("%s_vars.txt"%name).T
    er = np.sqrt(v)
    pd = (m-me)/me
    pde = er/me
    print m.shape, v.shape
    inds = np.arange(len(m[0]))
    import matplotlib.pyplot as plt
    labels = ['d0','d1','e0','e1','f0','f1','g0','g1']
    for i in range(len(m)):
        plt.errorbar(inds, m[i,:], er[i,:], label=labels[i])
    plt.xlim(-7,40)
    plt.xlabel("Simulation index")
    plt.legend(frameon=False, fontsize=10, loc='left')

    #for p, e in zip(pd, pde):
    #    plt.errorbar(inds, p, e)
    plt.show()

def make_corner(box):
    inpath = "./chain2_%s_box%d.txt"%(name, box)
    data = pd.read_csv(inpath, dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    data = data[nburn:]
    C = np.cov(data,rowvar=False)
    w,R = np.linalg.eig(C)
    print data.shape
    rD = np.dot(data[:],R) #Rotated data

    import corner
    import matplotlib.pyplot as plt
    fig = corner.corner(data)
    fig.savefig("cornertest%d.png"%box)
    plt.clf()

def corner_points():
    me = np.loadtxt("%s_medians.txt"%name).T
    m = np.loadtxt("%s_means.txt"%name).T
    v = np.loadtxt("%s_vars.txt"%name).T
    C = np.cov(m,rowvar=True)
    w, R = np.linalg.eig(C)
    print m.shape
    print R.shape
    m2 = np.dot(m.T, R)
    s = np.sqrt(v)
    s2 = np.dot(s.T, R)
    v2 = s2**2
    me2 = np.dot(me.T, R)
    print m2.shape
    print C.shape
    np.savetxt("r_special_medians.txt", me2)
    np.savetxt("r_special_means.txt", m2)
    np.savetxt("r_special_vars.txt", v2)
    np.savetxt("R_special.txt", R)
    m = m2.T
    #exit()
    Ns = len(m[0])
    Npars = len(m)
    N = Npars
    fig, axes = plt.subplots(N, N)
    labels = [r"$d_0$",r"$d_1$",r"$e_0$",r"$e_1$",r"$f_0$",r"$f_1$",r"$g_0$",r"$g_1$"]
    for i in range(0, N):
        for j in range(i+1, N):
            axes[i][j].remove()       
    for i in range(0,N):
        il = (min(m[i])-0.1, max(m[i])+0.1)
        for j in range(0,i+1):
            jl = (min(m[j])-0.1, max(m[j])+0.1)
            axes[i][j].set_xlim(jl)
            axes[i][j].set_ylim(il)
    for ind in range(0, Ns):
        for i in range(0,N):
            for j in range(0,i+1):
                axes[i][j].text(m[j,ind], m[i,ind], str(ind), color='k', fontsize=6)
    for i in range(N):
        axes[i][0].set_ylabel(labels[i])
        axes[-1][i].set_xlabel(labels[i])
    for i in range(1,N-1):
        for j in range(1,i+1):
            axes[i][j].set_yticklabels([])
            axes[i][j].set_xticklabels([])
    for j in range(1,N):
        axes[-1][j].set_yticklabels([])
    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.set_size_inches(8,8)
    fig.savefig("numbercorner.png")
    plt.show()
    print m.shape

if __name__ == "__main__":
    #make_means(40, 6)
    #print_means()
    #plot_means()
    #make_corner(0)
    #make_corner(2)
    #make_corner(10)
    #corner_points()
