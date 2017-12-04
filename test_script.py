import aemulus_data as AD
from classy import Class
from cluster_toolkit import massfunction
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

name = 'defg'

sfs = AD.get_scale_factors()
zs = 1./sfs - 1.
x = sfs - 0.5
volume = 1050.**3 #(Mpc/h)^3

def model_swap(params, name, xi):
    ints = params[::2]
    slopes = params[1::2]
    pars = ints + xi*slopes
    d, e, f, g = 1.97, 1.0, 0.51, 1.228
    if name == 'defg':
        d, e, f, g = pars
    if name == 'dfg':
        d, f, g = pars
    return d, e, f, g
    
def lnprior(params, args):
    x = args['x']
    ints = params[::2]
    slopes = params[1::2]
    for xi in x:
        pars = ints + xi*slopes
        if any(pars < 0) or any(pars > 5):
            return -np.inf
    return 0

def lnlike(params, args):
    x = args['x'] # a - 0.5
    Omega_m = args['Omega_m']
    M = args['M']
    k = args['k'] #h/Mpc
    ps = args['ps'] #(Mpc/h)^3
    Ns = args['Ns']
    edges = args['edges'] #Msun/h
    icovs = args['icovs']
    LL = 0
    for i in range(len(x)):
        d, e, f, g = model_swap(params, args['name'], x[i])
        dndM = massfunction.dndM_at_M(M, k , ps[i], Omega_m, d, e, f, g)
        N = massfunction.n_in_bins(edges[i], M, dndM)*volume
        X = Ns[i] - N
        LL += np.dot(X, np.dot(icovs[i], X))
        #pd = Ns[i]/N - 1.
        #plt.plot(pd)
    #plt.show()
    return -0.5*LL

def lnprob(params, args):
    lp = lnprior(params, args)
    if not np.isfinite(lp): return -1e22
    return lp + lnlike(params, args)

def get_cosmo(i):
    obh2, och2, w, ns, ln10As, H0, Neff, s8 = AD.get_building_box_cosmologies()[i]
    h = H0/100.
    Omega_b = obh2/h**2
    Omega_c = och2/h**2
    Omega_m = Omega_b+Omega_c
    params = {'output': 'mPk', 'h': h, 'ln10^{10}A_s': ln10As, 'n_s': ns, 'w0_fld': w, 'wa_fld': 0.0, 'Omega_b': Omega_b, 'Omega_cdm': Omega_c, 'Omega_Lambda': 1.- Omega_m, 'N_eff': Neff, 'P_k_max_1/Mpc':10., 'z_max_pk':10. }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo, h, Omega_m
    
def run_bf(i):
    Ns = []
    edges = []
    icovs = []
    cosmo, h, Omega_m = get_cosmo(i)
    #Get the cosmology here
    M = np.logspace(12, 16, num=200) #Msun/h
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    ps = []
    for j in range(0,10): #snap
        Mlo, Mhi, N, Mtot = AD.get_building_box_binned_mass_function(i, j).T
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.get_building_box_binned_mass_function_covariance(i, j)
        Ns.append(N)
        edges.append(edge)
        icovs.append(np.linalg.inv(cov))
        #Calculate each power spectrum at this redshift
        p = np.array([cosmo.pk_lin(ki, zs[j]) for ki in k])*h**3
        ps.append(p)
        
    args = {'x':x, 'k':k/h, 'ps':ps, 'edges':edges, 'Ns':Ns, 'icovs':icovs, 'Omega_m':Omega_m, 'h':h, 'M':M, 'name':name}

    #guess = np.array([ 0.98523686,  0.38452346,  0.88047635, -0.03644077,  1.13972382, -0.29261036])
    if name =='defg':
        #guess = np.array([2.13, 0.11, 1.1, 0.2, 0.41, 0.15, 1.25, 0.11])
        guess = np.array([2.347, 0.062, 1.040, 0.354, 0.451, 0.087, 1.288, 0.198])
    if name == 'dfg': guess = np.array([2.13, 0.11, 0.41, 0.15, 1.25, 0.11])
    print "Test lnprob call box%d: "%i, lnprob(guess, args)
    
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args, tol=1e-3)
    bfpath = "bf_%s_box%d.txt"%(args['name'], i)
    print result
    np.savetxt(bfpath, result.x)
    print "BF saved at: ",bfpath

def plot_bf(i):
    cosmo, h, Omega_m = get_cosmo(i)
    M = np.logspace(12, 16, num=200) #Msun/h
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    bfpath = "bf_%s_box%d.txt"%(name, i)
    print i, name
    params = np.loadtxt(bfpath)
    LL = 0
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(x))]
    fig, ax = plt.subplots(2, sharex=True)
    for j in range(0,10): #snap
        Mlo, Mhi, N, Mtot = AD.get_building_box_binned_mass_function(i, j).T
        Mave = Mtot/N
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.get_building_box_binned_mass_function_covariance(i, j)
        icov = np.linalg.inv(cov)
        err = np.sqrt(np.diag(cov))
        p = np.array([cosmo.pk_lin(ki, zs[j]) for ki in k])*h**3
        d, e, f, g = model_swap(params, name, x[j])
        dndM = massfunction.dndM_at_M(M, k/h, p, Omega_m, d, e, f, g)
        Nmodel = massfunction.n_in_bins(edge, M, dndM)*volume
        X = N-Nmodel
        chi2 = np.dot(X, np.dot(icov, X))
        print j, chi2
        LL += -0.5*chi2
        ax[0].errorbar(Mave, N, err, c=colors[j], marker='.', ls='')
        ax[0].loglog(Mave, Nmodel, ls='-', c=colors[j])
        pd = (Nmodel-N)/Nmodel
        pde = err/Nmodel
        ax[1].errorbar(Mave, pd, pde, c=colors[j])#, ls='')
    pdylim = .11
    ax[0].set_ylim(1, 1e6)
    ax[1].set_ylim(-pdylim, pdylim) 
    #plt.title(LL)
    plt.subplots_adjust(hspace=0)
    plt.show()

def plotpars():
    pars = []
    for i in range(0, 26):
        bfpath = "bf_%s_box%d.txt"%(name, i)
        pars.append(np.loadtxt(bfpath))
    pars = np.array(pars)
    print pars.shape
    labs = ['d0','d1','e0','e1','f0','f1','g0','g1']
    for i in range(len(pars[0])):
        plt.plot(pars[:,i], label=labs[i])
    means = np.mean(pars, 0)
    print means
    #for m in means:
    #    print "%.3f, "%m
    plt.legend(frameon=False)
    plt.show()
    
if __name__ == "__main__":
    #for i in xrange(26, 40):
    #    run_bf(i)
    #for i in xrange(0,26):
    #    plot_bf(i)
    plotpars()
