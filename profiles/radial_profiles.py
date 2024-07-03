# leer cat de Voids -> convertirlos en Void() -> leer cat de trazadores -> calcular perfiles -> apilarlos

import numpy as np
from astropy.io import fits
import time
import os

from tools.void import Void, Tracer
from tools.save_file import save_file
from tools.stacking_methods import parallel3, method3


def lens_cat(folder, lenses, 
             Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG):
    
    L = np.loadtxt(folder+lenses).T

    Rv    = L[1]
    z     = L[4]
    rho_1 = L[8] #Sobredensidad integrada a un radio de void 
    rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
    flag  = L[11]

    mask = ((Rv >= Rv_min)&(Rv < Rv_max))&((z >= z_min)&(z < z_max))&(
        (rho_1 >= rho1_min)&(rho_1 < rho1_max))&((rho_2 >= rho2_min)&(rho_2 < rho2_max))&(flag >= FLAG)        

    L = L[:,mask]

    return L

def main(tfolder, tracers, lfolder, lenses, sample,
         RMIN, RMAX, dr,
         Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG,
         ncores):

    print('Loading catalogs...')
    tracers = fits.open(tfolder+tracers)[1].data # catalog of tracers
    tmask = tracers.flag_central == 0
    ## setting the tcat to all voids instances
    Void.cat = tracers[tmask]
    del tracers

    L = lens_cat(lfolder, lenses, 
                 Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG)

    print(f'Nvoids: {len(L[1])}')
    
    t_in = time.time()

    print('Running stacking...')

    if ncores==1:
        stacked_profile = method3(xv=L[5], yv=L[6], zv=L[7], rv=L[1], RMIN=RMIN, RMAX=RMAX, dr=dr)
        print(f'Ended in {time.time()-t_in} s')
    else:
        stacked_profile = parallel3(ncores=ncores, L=L, RMIN=RMIN, RMAX=RMAX, dr=dr)
        print(f'Ended in {time.time()-t_in} s')

    #save files
    nvoids = len(L[1])

    folder = '../results/radial/'
    filename = f'sp-{sample}_r{int(Rv_min)}-{int(Rv_max)}_z0{int(z_min*10)}-0{int(z_max*10)}.fits'
    print(f'Saving file in {folder+filename}')
    s = save_file(folder, filename,
                  stacked_profile, nvoids, RMIN, RMAX, dr,
                  Rv_min, Rv_max, z_min, z_max)

    return s

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='pru')
    parser.add_argument('-lenses', action='store', dest='lenses',default='voids_MICE.dat')
    parser.add_argument('-tracers', action='store', dest='tracers',default='micecat2_halos_full.fits')

    parser.add_argument('-RMIN', action='store', dest='RMIN',default=0.01)
    parser.add_argument('-RMAX', action='store', dest='RMAX',default=5.0)
    parser.add_argument('-dr', action='store', dest='dr',default=0.25)
    
    parser.add_argument('-Rv_min', action='store', dest='Rv_min',default=6.)
    parser.add_argument('-Rv_max', action='store', dest='Rv_max',default=30.)
    parser.add_argument('-z_min', action='store', dest='z_min',default=0.1)
    parser.add_argument('-z_max', action='store', dest='z_max',default=0.4)
    parser.add_argument('-rho1_min', action='store', dest='rho1_min',default=-1.)
    parser.add_argument('-rho1_max', action='store', dest='rho1_max',default=1.)
    parser.add_argument('-rho2_min', action='store', dest='rho2_min',default=-1.)
    parser.add_argument('-rho2_max', action='store', dest='rho2_max',default=100.)

    parser.add_argument('-ncores', action='store', dest='ncores',default=32)

    args = parser.parse_args()

    sample = args.sample
    tracers = args.tracers
    lenses = args.lenses

    RMIN = float(args.RMIN)
    RMAX = float(args.RMAX)
    dr   = float(args.dr) # space between shells, in units of void radius
    
    Rv_min   = float(args.Rv_min)
    Rv_max   = float(args.Rv_max) 
    z_min    = float(args.z_min) 
    z_max    = float(args.z_max) 
    rho1_min = float(args.rho1_min)
    rho1_max = float(args.rho1_max) 
    rho2_min = float(args.rho2_min)
    rho2_max = float(args.rho2_max) 

    ncores = int(args.ncores)

    lfolder = '/mnt/simulations/MICE/' # folder of the lenses cat
    tfolder = '../../cats/MICE/'       # folder of the tracers cat

    FLAG = 2

    tini = time.time()
    S = main(tfolder=tfolder, tracers=tracers, lfolder=lfolder, lenses=lenses, sample=sample,
             RMIN=RMIN, RMAX=RMAX, dr=dr, 
             Rv_min=Rv_min, Rv_max=Rv_max, z_min=z_min, z_max=z_max, 
             rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2_min, rho2_max=rho2_max, FLAG=FLAG,
             ncores=ncores)
    tfin = time.time()
    if S:
        print(f'Radial profile finished succesfully in {tfin-tini} s!')
        print('--------')
