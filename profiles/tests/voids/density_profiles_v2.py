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

def density_v2(xh, yh, zh, lmhalo, N, m, rv, xv, yv, zv):
    N = int(N)
    m = int(m)
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

cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)
ra_gal, dec_gal, z_gal, lmhalo = tracercat_load()    
xh, yh, zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
print('# of gx: ', len(xh))

partial_func = partial(density_v2, xh, yh, zh, lmhalo)
def partial_func_unpack(A):    
    global xh, yh, zh, lmhalo
    return partial_func(*A)

def save_file(filename, lens_args, nvoids, N, m, P):

    name_args = ('Rv_min','Rv_max','z_min','z_max','rho1_min','rho1_max','rho2_min','rho_max')
    r_ad = np.linspace(0, m, N+1)
    r_ad = r_ad[:-1] + np.diff(r_ad)*0.5
    
    h = fits.Header()
    h.append(('Nvoids', int(nvoids)))
    
    for it,val in zip(name_args, lens_args):
        h.append((it, val))

    primary_hdu = fits.PrimaryHDU(header=h)
    hdul = fits.HDUList([primary_hdu])

    for p in P:
        r_v = (p[2][0]*3/(4*np.pi))**(1/3)*(N/m)
        table = np.array([fits.Column(name='r_phys', format='E', array=r_v*r_ad),
                          fits.Column(name='number', format='E', array=p[0]),
                          fits.Column(name='mass', format='E', array=p[1]),
                          fits.Column(name='vol_phys', format='E', array=p[2]),
                         ])
        tbhdu = fits.BinTableHDU.from_columns(table)
        hdul.append(tbhdu)
        
    hdul.writeto(filename, overwrite=True)
    
def main(lens_args=(6.0,9.0,0.2,0.3,-1.0,-0.8,0.0,100),
         ncores=32, N=10, m=5):

    L,_,nvoids = lenscat_load(*lens_args, 
                              flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat",
                              split=True, NSPLITS=ncores)
    
    print('# of voids: ',nvoids)
    
    P = np.zeros((nvoids, 4, N)) # 4=num de arr q devuelve density_v2
    for i,Li in enumerate(tqdm(L)):
        
        num = len(Li)
<<<<<<< HEAD
        print(num)
        entrada = np.array([Li.T[1], Li.T[5], Li.T[6], Li.T[7]]).T
        
=======
        entrada = np.array([np.full(num,N),
                            np.full(num,m),
                            Li.T[1], #rv
                            Li.T[5], #xv
                            Li.T[6], #yv
                            Li.T[7], #zv
                           ]).T
>>>>>>> cd8b6fcb3ca1ac1524872ff6a65323e90589f795
        with Pool(processes=num) as pool:
            resmap = pool.map(partial_func_unpack, entrada)
            pool.close()
            pool.join()
        for j,res in enumerate(resmap):
            P[i*num + j] = res
    
    if lens_args[7]<=0:
        t = 'R'
    elif lens_args[6]>=0:
        t = 'S'
    else:
        t = 'all'
    
    filename = f'{lensname.split("/")[-1][:-4]}_Rv{int(lens_args[0])}-{int(lens_args[1])}_z0{int(10*lens_args[2])}-0{int(10*lens_args[3])}_type-{t}.fits'
    print('Saving in '+filename)
    savefile(filename, lens_args, nvoids, N, m, P)
    print('End! :)')
    
if __name__ == '__main__':
    main(lens_args=(6.0,9.622,0.2,0.4,-1.0,-0.8,-1.0,100.0),ncores=32,N=50,m=5)