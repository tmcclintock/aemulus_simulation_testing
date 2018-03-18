import aemulus_data as AD
from classy import Class
from cluster_toolkit import massfunction, bias
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from scipy.special import comb
import emcee
import os, sys, itertools

sfs = AD.scale_factors()
zs = 1./sfs - 1.
x = sfs - 0.5
volume = 1050.**3 #(Mpc/h)^3

def model_swap(params, args, xi):
    dropped = args['dropped']
    kept = args['kept']

    pars = np.ones((8))
    pars[kept] = params
    if len(kept) != 8:
        defaults = args['defaults']
        pars[dropped] = defaults[dropped]
    #j = 0
    #for i in range(8):
    #    if i in dropped:
    #        pars[i] = defaults[i]
    #    else:
    #        pars[i] = params[j]
    #        j+=1
    d,e,f,g = pars[:4] + xi*pars[4:]
    return d, e, f, g
    
def lnprior(params, args):
    x = args['x']
    for xi in x:
        pars = np.array(model_swap(params, args, xi))
        if any(pars < 0) or any(pars > 5):
            return -np.inf
    return 0

def lnlike(params, args):
    x = args['x'] # a - 0.5
    Om = args['Omega_m']
    M = args['M']
    s2s = args['sigma2s']
    s2ts = args['sigma2tops']
    s2bs = args['sigma2bots']
    Ns = args['Ns'] #Number of halos in the sim
    edges = args['edges'] #Msun/h
    icovs = args['icovs']
    LL = 0
    for i in range(len(x)):
        d, e, f, g = model_swap(params, args, x[i])
        dndM = massfunction._dndM_sigma2_precomputed(M, s2s[i], s2ts[i], s2bs[i], Om, d, e, f, g)
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
    M = np.logspace(11.5, 16.5, num=200) #Msun/h
    Mt = M*(1-1e-6*0.5)
    Mb = M*(1+1e-6*0.5)
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    kh = k/h
    s2s = [] #sigma^2
    s2ts = [] #sigma^2 top part
    s2bs = [] #sigma^2 bottom part
    for j in range(0,10): #snap
        Mlo, Mhi, N, Mave = AD.building_box_binned_mass_function(i, j).T
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.building_box_binned_mass_function_covariance(i, j)
        #extra = np.diag((0.01*N)**2)
        #cov += extra
        Ns.append(N)
        edges.append(edge)
        icovs.append(np.linalg.inv(cov))
        #Calculate each power spectrum at this redshift
        p = np.array([cosmo.pk_lin(ki, zs[j]) for ki in k])*h**3
        s2s.append(bias.sigma2_at_M(M, kh, p, Omega_m))
        s2ts.append(bias.sigma2_at_M(Mt, kh, p, Omega_m))
        s2bs.append(bias.sigma2_at_M(Mb, kh, p, Omega_m))
    return {'x':x, 'edges':edges, 'Ns':Ns, 'icovs':icovs, 'Omega_m':Omega_m, 'h':h, 'M':M, 'sigma2s':s2s, 'sigma2tops':s2ts, 'sigma2bots':s2bs}
    
def run_bf(args, doprint=False):
    default_path = args['default_path']
    bfpath = args['bfpath']
    if os.path.isfile(default_path):
        guess = np.loadtxt(default_path)
    else:
        guess = np.array([1.97,1.0,0.51,1.1228,0.01,0.01,0.01,0.01])
    args['defaults'] = np.copy(guess)
    guess = guess[args['kept']]
    if doprint:
        print "Test lnprob() on box%d: \n\t%.3f"%(i, lnprob(guess, args))
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args)#, tol=1e-3)
    if doprint:
        print result
    np.savetxt(bfpath, result.x)
    if not os.path.isfile(default_path):
        np.savetxt(default_path, result.x)
    #print "BF saved at\n\t%s"%bfpath
    return result.fun

def run_mcmc(args, bfpath, mcmcpath, likespath):
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    nwalkers = 48
    nsteps = 2000
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

def plot_bf(i, args, bfpath, savepath=None):
    cosmo, h, Omega_m = get_cosmo(i)
    M = args['M']
    params = np.loadtxt(bfpath)
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(zs))]
    fig, ax = plt.subplots(2, sharex=True)
    for j in range(len(zs)):
        Mlo, Mhi, N, Mave = AD.building_box_binned_mass_function(i, j).T
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.building_box_binned_mass_function_covariance(i, j)
        err = np.sqrt(cov.diagonal())
        s2 = args['sigma2s'][j]
        s2t = args['sigma2tops'][j]
        s2b = args['sigma2bots'][j]
        d, e, f, g = model_swap(params, args, x[j])
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
    if savepath:
        plt.gcf().savefig(savepath)
    else:
        plt.show()
    
if __name__ == "__main__":
    inds = np.arange(8)
    nparams = [8,7,6,]
    for i in range(len(nparams)):
        npars = nparams[i]
        combos = itertools.combinations(inds, 8-npars)
        model_index = -1
        for combo in combos:
            model_index += 1
            if npars == 8:
                if model_index > 0:
                    continue
            if npars == 7:
                if model_index not in [5,6]:
                    continue
            if npars == 6:
                if model_index not in [3, 4, 5, 10, 15]:
                    continue
            lo = 10
            hi = lo+1
            ll = 0 #log likelihood
            for box in xrange(lo, hi):
                kept = np.delete(inds, combo)
                #print kept, combo, npars, model_index
                args = get_args(box)
                args['dropped'] = [ combo[k] for k in range(len(combo))]
                args['kept'] = kept
                args['name'] = "np%d_mi%d"%(npars,model_index)
                args['default_path'] = "defaults/defaults_np8_mi0.txt"
                bfpath = "bfs/bf_%s_box%d.txt"%(args['name'], box)
                args['bfpath'] = bfpath
                mcmcpath = "chains/chain_%s_box%d.txt"%(args['name'], box)
                likespath = "chains/likes_%s_box%d.txt"%(args['name'], box)
                ll += run_bf(args, doprint=False)
                plot_bf(box, args, bfpath, "figs/bf_%s_box%d.png"%(args['name'],box))
                #run_mcmc(args, bfpath, mcmcpath, likespath)
            print "Np%d Mi%d:\tlnlike = %e"%(npars, model_index, ll)

