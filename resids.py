import aemulus_data as AD
from classy import Class
from cluster_toolkit import massfunction
from cluster_toolkit import bias
import numpy as np
import matplotlib.pyplot as plt

name = 'defg'

sfs = AD.scale_factors()
zs = 1./sfs - 1.
x = sfs - 0.5
volume = 1050.**3 #(Mpc/h)^3

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

def get_box_resids(i):
    cosmo, h, Omega_m = get_cosmo(i)
    M = np.logspace(12, 16, num=200) #Msun/h
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    bfpath = "bf_%s_box%d.txt"%(name, i)
    print i, name
    params = np.loadtxt(bfpath)
    zb = np.array([])
    Mb = np.array([])
    nub = np.array([])
    R = np.array([])
    Re = np.array([])
    for j in range(0,10): #snap
        Mlo, Mhi, N, Mtot = AD.building_box_binned_mass_function(i, j).T
        Mj = Mtot/N
        Mb = np.concatenate((Mb, Mj))
        zb = np.concatenate((zb, np.ones_like(N)*zs[j]))
        edge = 10**np.concatenate((Mlo, Mhi[-1:]))
        cov = AD.building_box_binned_mass_function_covariance(i, j)
        icov = np.linalg.inv(cov)
        err = np.sqrt(np.diag(cov))
        p = np.array([cosmo.pk_lin(ki, zs[j]) for ki in k])*h**3
        nub = np.concatenate((nub, bias.nu_at_M(Mj, k/h, p, Omega_m)))
        d, e, f, g = model_swap(params, name, x[j])
        dndM = massfunction.dndM_at_M(M, k/h, p, Omega_m, d, e, f, g)
        Nmodel = massfunction.n_in_bins(edge, M, dndM)*volume
        R = np.concatenate((R, (Nmodel-N)/Nmodel))
        Re= np.concatenate((Re, err/Nmodel))
    return Mb, nub, zb, R, Re

def plot_resids():
    data = np.loadtxt("resid_out.txt")
    M,nu,z,R,Re = data.T
    plt.scatter(nu, R, marker='.', s=2)
    plt.ylim(-0.1, 0.1)
    xlim = plt.gca().get_xlim()
    plt.fill_between(xlim, -0.01, 0.01, color='gray', zorder=-1, alpha=0.2)
    plt.xlim(xlim)
    plt.ylabel("Fractional Error")
    plt.xlabel(r"$\nu$")
    plt.show()
    
if __name__ == "__main__":
    """
    M = np.array([])
    nu = np.array([])
    z = np.array([])
    R = np.array([])
    Re = np.array([])
    for i in xrange(0, 40):
        mb, nub, zb, rb, rbe = get_box_resids(i)
        M = np.concatenate((M, mb))
        nu = np.concatenate((nu, nub))
        z = np.concatenate((z, zb))
        R = np.concatenate((R, rb))
        Re = np.concatenate((Re, rbe))
    print M.shape, nu.shape, z.shape, R.shape, Re.shape
    output = np.array([M,nu,z,R,Re]).T
    np.savetxt("resid_out.txt", output)
    """
    plot_resids()
