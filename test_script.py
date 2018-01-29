import aemulus_data as AD
from classy import Class
from cluster_toolkit import massfunction
from cluster_toolkit import bias
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import emcee

name = 'special'
if '0' in name or '1' in name:
    symmetric = False
else:
    symmetric = True #A symmetric model, with both slopes and intercepts
sfs = AD.scale_factors()
zs = 1./sfs - 1.
x = sfs - 0.5
volume = 1050.**3 #(Mpc/h)^3

R_special = np.loadtxt("chains/R_special.txt")

def model_swap(params, name, xi):
    d, e, f, g = 1.97, 1.0, 0.51, 1.228
    if name == 'special': #work in the rotated space
        p2 = np.dot(R_special, params)
        ints = p2[::2]
        slopes = p2[1::2]
        return ints + xi*slopes
    if symmetric:
        ints = params[::2]
        slopes = params[1::2]
        pars = ints + xi*slopes
        if name == 'defg':
            d, e, f, g = pars
        if name == 'dfg':
            d, f, g = pars
        if name == 'efg':
            e, f, g = pars
    if name == 'def0g':
        d0,d1,e0,e1,f0,g0,g1 = params
        d, e, g = d0+xi*d1, e0+xi*e1, g0+xi*g1
        f = f0+xi*0.092
    if name == 'de0fg':
        d0,d1,e0,f0,f1,g0,g1 = params
        d, f, g = d0+xi*d1, f0+xi*f1, g0+xi*g1
        e = e0+xi*0.3098
    if name == 'def1g':
        d0,d1,e0,e1,f1,g0,g1 = params
        d, e, g = d0+xi*d1, e0+xi*e1, g0+xi*g1
        f = 0.4384+xi*f1
    if name == 'de1fg':
        d0,d1,e1,f0,f1,g0,g1 = params
        d, f, g = d0+xi*d1, f0+xi*f1, g0+xi*g1
        e = 1.11+xi*e1
    return d, e, f, g
    
def lnprior(params, args):
    x = args['x']
    for xi in x:
        pars = np.array(model_swap(params, args['name'], xi))
        if any(pars < 0) or any(pars > 5):
            return -np.inf
        if args['name'] == 'special':
            g0,g1 = params[-2:]
            lpg0 = -0.5*(g0-0.815)**2/0.0156
            lpg1 = -0.5*(g1-0.512)**2/0.00102
            return lpg0+lpg1
    return 0

def lnlike(params, args):
    x = args['x'] # a - 0.5
    Omega_m = args['Omega_m']
    M = args['M']
    s2s = args['sigma2s']
    s2ts = args['sigma2tops']
    s2bs = args['sigma2bots']
    Ns = args['Ns'] #Number of halos in the sim
    edges = args['edges'] #Msun/h
    icovs = args['icovs']
    LL = 0
    for i in range(len(x)):
        d, e, f, g = model_swap(params, args['name'], x[i])
        dndM = massfunction._dndM_sigma2_precomputed(M, s2s[i], s2ts[i], s2bs[i], Omega_m, d, e, f, g)
        N = massfunction.n_in_bins(edges[i], M, dndM)*volume
        X = Ns[i] - N
        LL += np.dot(X, np.dot(icovs[i], X))
    return -0.5*LL

def lnprob(params, args):
    lp = lnprior(params, args)
    if not np.isfinite(lp): return -1e22
    return lp + lnlike(params, args)

def get_cosmo(i):
    obh2, och2, w, ns, ln10As, H0, Neff, s8 = AD.building_box_cosmologies()[i]
    h = H0/100.
    Omega_b = obh2/h**2
    Omega_c = och2/h**2
    Omega_m = Omega_b+Omega_c
    params = {'output': 'mPk', 'h': h, 'ln10^{10}A_s': ln10As, 'n_s': ns, 'w0_fld': w, 'wa_fld': 0.0, 'Omega_b': Omega_b, 'Omega_cdm': Omega_c, 'Omega_Lambda': 1.- Omega_m, 'N_eff': Neff, 'P_k_max_1/Mpc':10., 'z_max_pk':10. }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo, h, Omega_m

def get_args(i):
    Ns = []
    edges = []
    icovs = []
    cosmo, h, Omega_m = get_cosmo(i)
    M = np.logspace(12, 16, num=200) #Msun/h
    Mt = M*(1-1e-6*0.5)
    Mb = M*(1+1e-6*0.5)
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    kh = k/h
    s2s = [] #sigma^2
    s2ts = [] #sigma^2 top part
    s2bs = [] #sigma^2 bottom part
    for j in range(0,10): #snap
        Mlo, Mhi, N, Mtot = AD.building_box_binned_mass_function(i, j).T
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.building_box_binned_mass_function_covariance(i, j)
        Ns.append(N)
        edges.append(edge)
        icovs.append(np.linalg.inv(cov))
        #Calculate each power spectrum at this redshift
        p = np.array([cosmo.pk_lin(ki, zs[j]) for ki in k])*h**3
        s2s.append(bias.sigma2_at_M(M, kh, p, Omega_m))
        s2ts.append(bias.sigma2_at_M(Mt, kh, p, Omega_m))
        s2bs.append(bias.sigma2_at_M(Mb, kh, p, Omega_m))
    return {'x':x, 'edges':edges, 'Ns':Ns, 'icovs':icovs, 'Omega_m':Omega_m, 'h':h, 'M':M, 'name':name, 'sigma2s':s2s, 'sigma2tops':s2ts, 'sigma2bots':s2bs}
    
def run_bf(args, bfpath):
    if name == 'special':
        guess = np.array([0.339,1.058,2.468,-0.221,0.141,0.253,0.815,0.512]) #defg rotated
    if name =='defg':
        guess = np.array([2.287, 0.039, 1.11, 0.31, 0.44, 0.092, 1.27, 0.162]) #defg
    if name == 'de0fg':
        guess = np.array([2.287, 0.039, 1.11, 0.44, 0.092, 1.27, 0.162]) #de0fg, e1 is off
    if name == 'def0g':
        guess = np.array([2.287, 0.039, 1.11, 0.31, 0.44, 1.27, 0.162]) #def0g, f1 is off
    if name =='de1fg':
        guess = np.array([2.287, 0.039, 0.31, 0.44, 0.092, 1.27, 0.162]) #de1fg, e0 is off
    if name =='def1g':
        guess = np.array([2.287, 0.039, 1.11, 0.31, 0.092, 1.27, 0.162]) #def1g, f0 is off
    if name == 'dfg':
        guess = np.array([2.13, 0.11, 0.41, 0.15, 1.25, 0.11]) #dfg
    if name == 'efg':
        guess = np.array([1.040, 0.354, 0.451, 0.087, 1.288, 0.198]) #efg

    print "Test lnprob() on box%d: \n\t%.3f"%(i, lnprob(guess, args))

    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args)#, tol=1e-3)
    print result
    np.savetxt(bfpath, result.x)
    print "BF saved at\n\t%s"%bfpath
    return

def run_mcmc(args, bfpath, mcmcpath, likespath):
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    nwalkers = 48
    nsteps = 10000
    pos = [bf + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=4)
    print "Running MCMC for model:\n\t%s"%(args['name'])
    print "Using fits from:\n\t%s"%bfpath
    sampler.run_mcmc(pos, nsteps)
    print "Saving chain at:\n\t%s"%mcmcpath
    chain = sampler.flatchain
    np.savetxt(mcmcpath, chain)
    likes = sampler.flatlnprobability
    np.savetxt(likespath, likes)

def plot_bf(i, args, bfpath):
    cosmo, h, Omega_m = get_cosmo(i)
    M = args['M']
    params = np.loadtxt(bfpath)
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(zs))]
    fig, ax = plt.subplots(2, sharex=True)
    for j in range(len(zs)):
        Mlo, Mhi, N, Mtot = AD.building_box_binned_mass_function(i, j).T
        Mave = Mtot/N #Average mass of halos in each bin
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.building_box_binned_mass_function_covariance(i, j)
        err = np.sqrt(cov.diagonal())
        s2 = args['sigma2s'][j]
        s2t = args['sigma2tops'][j]
        s2b = args['sigma2bots'][j]
        d, e, f, g = model_swap(params, name, x[j])
        dndM = massfunction._dndM_sigma2_precomputed(M, s2, s2t, s2b, Omega_m, d, e, f, g)
        Nmodel = massfunction.n_in_bins(edge, M, dndM)*volume
        chi2 = np.dot(N-Nmodel, np.dot(np.linalg.inv(cov), N-Nmodel))
        print "Box%d snap%d chi2 = %.2f"%(i, j, chi2)
        ax[0].errorbar(Mave, N, err, c=colors[j], marker='.', ls='')
        ax[0].loglog(Mave, Nmodel, ls='-', c=colors[j])
        pd = (Nmodel-N)/Nmodel
        pde = err/Nmodel
        ax[1].errorbar(Mave, pd, pde, c=colors[j])
    ax[0].set_ylim(1, 1e6)
    yl = .11
    ax[1].set_ylim(-yl, yl) 
    plt.subplots_adjust(hspace=0)
    plt.show()
    
if __name__ == "__main__":
    lo = 0
    hi = 1
    for i in xrange(lo, hi):
        args = get_args(i)
        bfpath = "bfs/bf_%s_box%d.txt"%(args['name'], i)
        mcmcpath = "chains/chain2_%s_box%d.txt"%(args['name'], i)
        likespath = "chains/likes2_%s_box%d.txt"%(args['name'], i)
        run_bf(args, bfpath)
        plot_bf(i, args, bfpath)
        #run_mcmc(args, bfpath, mcmcpath, likespath)
