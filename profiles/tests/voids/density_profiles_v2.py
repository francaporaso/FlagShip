import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from astropy.cosmology import LambdaCDM
from astropy.io import fits
import sys
sys.path.append('/home/fcaporaso/FlagShip/profiles/')
sys.path.append('/home/fcaporaso/FlagShip/vgcf/')
from perfiles import lenscat_load
from vgcf import ang2xyz
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def tracercat_load(catname='mice_sats_18939.fits', if_centrals=True):
    
    with fits.open('/home/fcaporaso/cats/MICE/'+catname) as f:
        ra_gal  = f[1].data.ra_gal
        dec_gal = f[1].data.dec_gal
        z_gal   = f[1].data.z_cgal
        centrals = f[1].data.flag_central == 0
        lmhalo = f[1].data.lmhalo
        
    if if_centrals:
        return ra_gal[centrals], dec_gal[centrals], z_gal[centrals], lmhalo[centrals]
    
    return ra_gal, dec_gal, z_gal

def number_density_v2(N, m, xh, yh, zh, lmhalo, rv, xv, yv, zv):
    number_gx = np.zeros(N)
    mass_bin = np.zeros(N)
    vol = np.zeros(N)
    dist = np.sqrt((xh-xv)**2 + (yh-yv)**2 + (zh-zv)**2) ## dist to center of void i
    const = m*rv/N

    for k in range(N):
        mask = (dist < (k+1)*const) & (dist >= k*const)
        number_gx[k] = mask.sum()
        mass_bin[k] = np.sum( 10.0**(lmhalo[mask]) )
        vol[k] = (k+1)**3 - k**3

    vol *= (4/3)*np.pi*const**3
    
    return number_gx, mass_bin, vol

partial_func = partial(number_density_v2, N, m, xh, yh, zh, lmhalo)

def partial_func_unpack(A):
    return partial_func(*A)

def main(lens_args=(6.0,9.0,0.2,0.3,-1.0,-0.8,0.0,100),
         ncores=32, N=10, m=5):

    cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

    L,_,nvoids = lenscat_load(*lens_args, 
                         flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat",
                         split=True, NSPLITS=ncores)
    print('# of voids: ',nvoids)
    
    ra_gal, dec_gal, z_gal, lmhalo = tracercat_load()    
    print('# of gx: ', len(ra_gal))
    
    ra_gal, dec_gal, z_gal, lmhalo = tracercat_load()
    xh, yh, zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
    
    for i,Li in enumerate(tqdm(L)):
        with Pool(processes=32) as pool:
            entrada = np.array([Li.T[1], Li.T[5], Li.T[6], Li.T[7]]).T
            res = pool.map(partial_func_unpack,
                           entrada)
            pool.close()
            pool.join()
    
    
if __name__ == '__main__':
    main()