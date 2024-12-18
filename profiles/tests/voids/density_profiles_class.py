from astropy.cosmology import LambdaCDM
from astropy.io import fits
# from functools import partial
# import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import sys
from tqdm import tqdm
#sys.path.append('/home/fcaporaso/FlagShip/profiles/')
sys.path.append('/home/fcaporaso/FlagShip/vgcf/')
from vgcf import ang2xyz

class Void:

    def __init__(self, lensname='/mnt/simulations/MICE/voids_MICE.dat',
                 lens_args={'Rv_min':6.0,'Rv_max':9.0,
                            'z_min':0.2,'z_max':0.4,
                            'rho1_min':-1.0,'rho1_max':-0.8,
                            'rho2_min':-1.0,'rho2_max':100.0},
                 catname='/home/fcaporaso/cats/MICE/mice_sats_18939.fits',
                 split=True, if_centrals=True,
                 N=10, m=5, ncores=4):
        
        self.cosmo : LambdaCDM = LambdaCDM(H0=100.0, Om0=0.25, Ode0=0.75)

        self.N      : int = N
        self.m      : int = m
        self.ncores : int = ncores
        
        self.lensname  : str        = lensname
        self.lens_args : dict       = lens_args
        self.split     : bool       = split
        self.voidcat   : np.ndarray = None
        self.nvoids    : int        = 0
        self.Kmask     : np.ndarray = None

        self.catname    : str        = catname
        self.if_centrals: bool       = if_centrals
        self.xh         : np.ndarray = None
        self.yh         : np.ndarray = None
        self.zh         : np.ndarray = None
        self.lmhalo     : np.ndarray = None
        self.ngx        : int        = 0

    def load_voidcat(self):
        '''
        loads void catalog splited for multiprocessing
        '''
        ## 0:id, 1:Rv, 2:ra, 3:dec, 4:z, 5:xv, 6:yv, 7:zv, 8:rho1, 9:rho2, 10:logp, 11:flag
        L = np.loadtxt(self.lensname).T

        nk = 100 ## para cambiarlo hay que repensar el calculo de (dra,ddec) y el doble for loop
        NNN = len(L[0]) ##total number of voids
        ra,dec = L[2],L[3]
        K    = np.zeros((nk+1,NNN))
        K[0] = np.ones(NNN).astype(bool)

        ramin  = np.min(ra)
        cdec   = np.sin(np.deg2rad(dec))
        decmin = np.min(cdec)
        dra    = ((np.max(ra)+1.e-5) - ramin)/10.
        ddec   = ((np.max(cdec)+1.e-5) - decmin)/10.

        c = 1
        for a in range(10): 
            for d in range(10): 
                mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
                mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
                K[c] = ~(mra&mdec)
                c += 1

        mask = (L[1] >= self.lens_args['Rv_min']) & (L[1] < self.lens_args['Rv_max']) & (
            L[4] >= self.lens_args['z_min']) & (L[4] < self.lens_args['z_max']) & (
            L[8] >= self.lens_args['rho1_min']) & (L[8] < self.lens_args['rho1_max']) & (
            L[9] >= self.lens_args['rho2_min']) & (L[9] < self.lens_args['rho2_max']) & (
            L[11] >= 2.0)

        nvoids = mask.sum()
        L = L[:,mask]

        if self.split:
            if self.ncores > nvoids:
                self.ncores = nvoids
            lbins = int(round(nvoids/float(self.ncores), 0))
            slices = ((np.arange(lbins)+1)*self.ncores).astype(int)
            slices = slices[(slices < nvoids)]
            L = np.split(L.T, slices)
            K = np.split(K.T, slices)

        self.voidcat = L
        self.nvoids = nvoids
        self.Kmask =  K

    def load_gxcat(self):

        if self.if_centrals:    
            with fits.open(self.catname) as f:
                centrals = f[1].data.flag_central == 0
                ra_gal  = f[1].data.ra_gal[centrals]
                dec_gal = f[1].data.dec_gal[centrals]
                z_gal   = f[1].data.z_cgal[centrals]
                lmhalo  = f[1].data.lmhalo[centrals]
            
            self.xh, self.yh, self.zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=self.cosmo)
            self.lmhalo = lmhalo
            self.ngx = len(self.xh)

        else:
            with fits.open(self.catname) as f:
                ra_gal  = f[1].data.ra_gal
                dec_gal = f[1].data.dec_gal
                z_gal   = f[1].data.z_cgal
            
            self.xh, self.yh, self.zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=self.cosmo)
            self.ngx = len(self.xh)

    def partial_density(self, params):
        
        rv, xv, yv, zv = params
        
        number_gx = np.zeros(self.N)
        mass_bin = np.zeros(self.N)
        vol = np.zeros(self.N)
        
        dist = np.sqrt((self.xh-xv)**2 + (self.yh-yv)**2 + (self.zh-zv)**2) ## dist to center of void i
        const = self.m*rv/self.N

        mask_mean = (dist < 5*self.m*rv)
        dist = dist[mask_mean]
        lmhalo = self.lmhalo[mask_mean]
        
        mass_ball = np.sum( 10.0**(lmhalo) )
        mean_den_ball = mass_ball/((4/3)*np.pi*(5*self.m*rv)**3)

        for k in range(self.N):
            mask = (dist < (k+1)*const) & (dist >= k*const)
            number_gx[k] = mask.sum()
            mass_bin[k] = np.sum( 10.0**(lmhalo[mask]) )
            vol[k] = (k+1)**3 - k**3
            
        vol *= (4/3)*np.pi*const**3
        
        return number_gx, mass_bin, vol, np.full_like(vol, mean_den_ball)
    
    def stacking(self):
        P = np.zeros((self.nvoids, 4, self.N)) # 4=num de arr q devuelve density_v2
    
        for i,Li in enumerate(tqdm(self.voidcat)):
            
            num = len(Li)
            entrada = np.array([Li.T[1], #rv
                                Li.T[5], #xv
                                Li.T[6], #yv
                                Li.T[7], #zv
                               ]).T
            with Pool(processes=num) as pool:
                resmap = pool.map(self.partial_density, entrada)
                pool.close()
                pool.join()

            P[i*num:(i+1)*num] = resmap
        
        return P