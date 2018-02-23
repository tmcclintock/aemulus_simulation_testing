"""
Take some results, such as efg, and rotate the chains to break tight correlations.
"""
import numpy as np
import corner, sys, os
import matplotlib.pyplot as plt

name = "de1fg"
if name == 'defg': Np = 8
if name == 'dfg': Np = 6
if '0' in name or '1' in name: Np = 7

#Just use Box016 to find the rotations
base_dir = "../chains/"
inbase = base_dir+"/chain2_"+name+"_box%d.txt"
base_save = "./"
chainout = base_save+"/rotated_"+name+"_box%d_chain.txt"
Rout     = base_save+"/R_"+name+"_box%d.txt"

nw = 48 #number of walkers
nburn = 1000 #steps/walker to burn

def make_Rs(i):
    data = np.loadtxt(inbase%i)[nw*nburn:]
    D = np.copy(data)
    C = np.cov(D,rowvar=False)
    w,Ri = np.linalg.eig(C)
    np.savetxt(Rout%i,Ri)
    print "Saved R at :\n\t%s"%(Rout%i)
    
#As it turns out, cosmo 34 is the middle-most box,
#so use it for the final rotation matrix.
def make_Rs_and_rotate(i, R_ind=34):
    R = np.loadtxt(Rout%R_ind)
    #print "Using R at :\n\t%s"%(Rout%R_ind)
    data = np.loadtxt(inbase%i)[nw*nburn:]
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

def make_means_and_vars(N, Np):
    median_models = np.zeros((N, Np))
    mean_models = np.zeros((N, Np))
    var_models  = np.zeros((N, Np))
    for i in range(0, N):
        data = np.loadtxt(chainout%i)
        median_models[i] = np.median(data, 0)
        mean_models[i] = np.mean(data, 0)
        var_models[i] = np.var(data, 0)
        print "Means calculated for %d"%i
    np.savetxt("r_%s_medians.txt"%name, median_models)
    np.savetxt("r_%s_means.txt"%name, mean_models)
    np.savetxt("r_%s_vars.txt"%name,  var_models)

def make_means_nonrot(N, Np):
    median_models = np.zeros((N, Np))
    mean_models = np.zeros((N, Np))
    var_models  = np.zeros((N, Np))
    for i in range(0, N):
        data = np.loadtxt(inbase%i)
        median_models[i] = np.median(data, 0)
        mean_models[i] = np.mean(data, 0)
        var_models[i] = np.var(data, 0)
        print "Means calculated for %d"%i
    np.savetxt("%s_medians.txt"%name, median_models)
    np.savetxt("%s_means.txt"%name, mean_models)
    np.savetxt("%s_vars.txt"%name,  var_models)

def plot_means():
    m = np.loadtxt("%s_means.txt"%name).T
    v = np.loadtxt("%s_vars.txt"%name).T
    er = np.sqrt(v)
    i = np.arange(len(m[0]))
    j = 0
    for mi, ei in zip(m, er):
        plt.errorbar(i, mi, ei,label="%d"%j)
        j+=1
        print np.mean(mi)
    plt.legend(frameon=False)
    plt.show()

if __name__ == "__main__":
    #make_Rs(34)
    #make_means_nonrot(40, Np)
    #for i in xrange(34,35):#0,40):
    #    make_Rs_and_rotate(i, R_ind=34)
    #make_means_and_vars(40, Np)
    plot_corners(34)
    #plot_means()
