"""
Take some results, such as efg, and rotate the chains to break tight correlations.
"""
import numpy as np
import pandas as pd
import corner, sys, os
import matplotlib.pyplot as plt

name = "de1fg"
if name == 'defg': Np = 8
if name == 'dfg': Np = 6
if '0' in name or '1' in name: Np = 7

use1p = True

#Just use Box016 to find the rotations
base_dir = "../chains/"
if use1p:
    base_dir+="half_percent_chains/"
inbase = base_dir+"/chain2_"+name+"_box%d.txt"
base_save = "./"
chainout = base_save+"/rotated_"+name+"_box%d_chain.txt"
Rout     = base_save+"/R_"+name+"_box%d.txt"
rotate_matrix_path = base_save+"R_"+name+".txt"

nw = 48 #number of walkers
nburn = 1000 #steps/walker to burn

def make_rotation_matrix(i):
    data = pd.read_csv(inbase%i, dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    data = data[nw*nburn:]
    D = np.copy(data)
    C = np.cov(D,rowvar=False)
    w,Ri = np.linalg.eig(C)
    np.savetxt(rotate_matrix_path,Ri)
    print "Saved R at :\n\t%s"%(rotate_matrix_path)

#As it turns out, cosmo 34 is the middle-most box,
#so use it for the final rotation matrix.
def make_Rs_and_rotate(i, R_ind=34):
    R = np.loadtxt(rotate_matrix_path)
    print "Using R at:\n\t%s"%(rotate_matrix_path)
    print "Reading chain from:\n\t%s"%(inbase%i)
    data = pd.read_csv(inbase%i, dtype='float64', delim_whitespace=True)
    data = data.as_matrix()
    data = data[nw*nburn:]
    C = np.cov(data,rowvar=False)
    w,Ri = np.linalg.eig(C)
    rD = np.dot(data[:],R) #Rotated data
    np.savetxt(Rout%i,Ri)
    print "Created R%d"%i
    np.savetxt(chainout%i,rD)
    print "Saved rotated chain %d"%i, rD.shape
    
def plot_corners(i):
    data = np.loadtxt(inbase%i)
    l = len(data)
    data = data[int(0*l):]
    import corner as corner
    fig = corner.corner(data)
    import matplotlib.pyplot as plt
    fig.savefig("corner_nonrotated.png")
    #plt.show()
    plt.clf()
    data = np.loadtxt(chainout%i)
    l = len(data)
    data = data[int(0*l):]
    fig = corner.corner(data)
    import matplotlib.pyplot as plt
    fig.savefig("corner_rotated.png")
    #plt.show()

def make_means_and_vars(N, Np, name):
    median_models = np.zeros((N, Np))
    mean_models = np.zeros((N, Np))
    var_models  = np.zeros((N, Np))
    for i in range(0, N):
        data = pd.read_csv(chainout%i, dtype='float64', delim_whitespace=True)
        data = data.as_matrix()
        median_models[i] = np.median(data, 0)
        mean_models[i] = np.mean(data, 0)
        var_models[i] = np.var(data, 0)
        print "Means calculated for %d"%i
    p1= ""
    if use1p:
        p1 = "hp"
    np.savetxt(p1+"r_%s_medians.txt"%name, median_models)
    np.savetxt(p1+"r_%s_means.txt"%name, mean_models)
    np.savetxt(p1+"r_%s_vars.txt"%name,  var_models)
    print "Means saved at:\n\t%s"%(p1+"r_%s_means.txt"%name)

def make_means_nonrot(N, Np, name):
    median_models = np.zeros((N, Np))
    mean_models = np.zeros((N, Np))
    var_models  = np.zeros((N, Np))
    for i in range(0, N):
        data = pd.read_csv(inbase%i, dtype='float64', delim_whitespace=True)
        data = data.as_matrix()
        median_models[i] = np.median(data, 0)
        mean_models[i] = np.mean(data, 0)
        var_models[i] = np.var(data, 0)
        print "Means calculated for %d"%i
    if use1p:
        name = "hp"+name
    np.savetxt("%s_medians.txt"%name, median_models)
    np.savetxt("%s_means.txt"%name, mean_models)
    np.savetxt("%s_vars.txt"%name,  var_models)

def plot_means():
    m = np.loadtxt("r_%s_means.txt"%name).T
    v = np.loadtxt("r_%s_vars.txt"%name).T
    er = np.sqrt(v)
    i = np.arange(len(m[0]))
    j = 0
    for mi, ei in zip(m, er):
        plt.errorbar(i, mi, ei,label="%d"%j)
        j+=1
        print np.mean(mi)
    plt.legend(frameon=False)
    plt.show()

def make_real_corner(i):
    chain = pd.read_csv(chainout%i, dtype='float64', delim_whitespace=True)
    chain = chain.as_matrix()
    from chainconsumer import ChainConsumer
    labs=[r"$d_0'$",r"$d_1'$",r"$e_1'$",r"$f_0'$",r"$f_1'$",r"$g_0'$",r"$g_1'$"]
    c = ChainConsumer()
    c.add_chain(chain, parameters=labs)
    c.configure(kde=True, tick_font_size=10, label_font_size=24, max_ticks=3, sigmas=[0,1,2,3], usetex=True)#, statistics='max_symmetric')
    fig = c.plotter.plot()#legend=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig("fig_Rcorner.pdf", bbox_inches='tight')
    plt.show()    

    
if __name__ == "__main__":
    make_rotation_matrix(34)
    #make_means_nonrot(40, Np, name)
    for i in xrange(0,40):
        make_Rs_and_rotate(i, R_ind=34)
    make_means_and_vars(40, Np, name)
    #plot_corners(23)
    #plot_means()
    #make_real_corner(np.random.randint(40))
