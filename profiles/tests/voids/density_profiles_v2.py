from astropy.cosmology import LambdaCDM
from astropy.io import fits
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from pandas import DataFrame
import sys
from tqdm import tqdm
sys.path.append('/home/fcaporaso/FlagShip/profiles/')
sys.path.append('/home/fcaporaso/FlagShip/vgcf/')
from perfiles import lenscat_load
from vgcf import ang2xyz

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

def density_v2(N, m, xh, yh, zh, lmhalo, void_prop):
    rv = void_prop[0]
    xv = void_prop[1]
    yv = void_prop[2]
    zv = void_prop[3]
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
        
    mask_mean = (dist < 5*m*const)
    mass_ball = np.sum( 10.0**(lmhalo[mask_mean]) )
    mean_den_ball = mass_ball/((4/3)*np.pi*(5*m*const)**3)
    
    vol *= (4/3)*np.pi*const**3
    
    return number_gx, mass_bin, vol, np.full_like(vol, mean_den_ball)

def main(lens_args=(6.0,9.0,0.2,0.3,-1.0,-0.8,0.0,100),
         ncores=32, N=10, m=5):
    
    cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

    L,_,nvoids = lenscat_load(*lens_args, 
                              flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat",
                              split=True, NSPLITS=ncores)
    
    print('# of voids: ',nvoids)
    
    ra_gal, dec_gal, z_gal, lmhalo = tracercat_load()    
    xh, yh, zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
    print('# of gx: ', len(ra_gal))
    
    ## func for paralellization, returns func w 
    partial_func = partial(density_v2, N, m, xh, yh, zh, lmhalo)
    
    P = np.zeros((nvoids, 4, N)) # 4=num de arr q devuelve density_v2
    for i,Li in enumerate(tqdm(L)):
        
        num = len(Li)
        print(num)
        entrada = np.array([Li.T[1], Li.T[5], Li.T[6], Li.T[7]]).T
        
        with Pool(processes=num) as pool:
            resmap = pool.map(partial_func,
                              entrada)
            pool.close()
            pool.join()
        for j,res in enumerate(resmap):
            P[i*num + j] = res

    return P
    
if __name__ == '__main__':
    main()