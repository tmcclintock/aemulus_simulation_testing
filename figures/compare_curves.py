import matplotlib.pyplot as plt
import aemulus_data as AD
from classy import Class
import numpy as np
from cluster_toolkit import massfunction

name = 'defg'

sfs = AD.scale_factors()
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

def mass_func(pars, args):
    x = args['x']
    Omega_m = args['Omega_m']
    M = args['M']
    k = args['k']
    ps = args['ps']
    edges = args['edges'] #Msun/h
    Nout = []
    for i in range(len(x)):
        d, e, f, g = model_swap(pars, args['name'], x[i])
        dndM = massfunction.dndM_at_M(M, k , ps[i], Omega_m, d, e, f, g)
        Nout.append(massfunction.n_in_bins(edges[i], M, dndM)*volume)
    return Nout

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
    errs = []
    cosmo, h, Omega_m = get_cosmo(i)
    #Get the cosmology here
    M = np.logspace(12, 16, num=200) #Msun/h
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    ps = []
    Mxs = []
    for j in range(0,10): #snap
        Mlo, Mhi, N, Mtot = AD.building_box_binned_mass_function(i, j).T
        cov = AD.building_box_binned_mass_function_covariance(i, j)
        errs.append(np.sqrt(cov.diagonal()))
        Mx = Mtot/N
        Mxs.append(Mx)
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        Ns.append(N)
        edges.append(edge)
        #Calculate each power spectrum at this redshift
        p = np.array([cosmo.pk_lin(ki, zs[j]) for ki in k])*h**3
        ps.append(p)
    return {'x':x, 'k':k/h, 'ps':ps, 'edges':edges, 'Ns':Ns, 'Omega_m':Omega_m, 'h':h, 'M':M, 'name':name, 'Mx':Mxs, 'errs':errs}

def get_bf(i, args):
    inpath = "../bfs/bf_%s_box%d.txt"%(args['name'], i)
    return np.loadtxt(inpath)

def get_mc(i, args):
    inpath = "../chains/%s_means.txt"%args['name']
    return np.loadtxt(inpath)[i]

def get_Rmc(i, args):
    inpath = "../diag_chains/r_%s_means.txt"%args['name']
    R = np.loadtxt("../diag_chains/R_%s_box34.txt"%args['name'])
    pars = np.loadtxt(inpath)[i]
    return np.dot(R, pars).flatten()

def get_emu(i, args):
    return np.array([2.19935377,-0.34132756, 1.08308698, 0.32142226, 0.42534707, 0.06507512,1.22173862,-0.02819535])

if __name__ == "__main__":
    ind = 36

    
    args = get_args(ind)
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(x))]
    fig, ax = plt.subplots(2, sharex=True)

    for k in range(4):
        if k == 0:
            params = get_bf(ind, args)
            print params
            mark = '^'
        if k == 1:
            params = get_mc(ind, args)
            print params
            mark = 'o'
        if k == 2:
            params = get_Rmc(ind, args)
            print params
            mark = '<'
            continue
        if k == 3:
            params = get_emu(ind, args)
            print params
            mark = 'D'
        Nouts = mass_func(params, args)
        for i in range(0, 10): #snap
            if i < 8: continue
            Nout = Nouts[i]
            Mx = args['Mx'][i]
            N = args['Ns'][i]
            err = args['errs'][i]
            ax[0].errorbar(Mx, N, err, c=colors[i], marker='o', ls='', ms=1)
            ax[0].loglog(Mx, Nout, ls='-', c=colors[i])
            pd = (Nout-N)/Nout
            pde = err/Nout
            #ax[1].errorbar(Mx, pd, pde, c=colors[i], ls='-')#, marker='^')
            ax[1].plot(Mx, pd, c=colors[i], ls='', marker=mark, markersize=1.5)
        
    pdylim = .11
    ax[0].set_ylim(1, 1e6)
    ax[1].set_ylim(-pdylim, pdylim) 
    #plt.title(LL)
    plt.subplots_adjust(hspace=0)
    ax[1].set_xlabel("Mass")
    plt.show()

