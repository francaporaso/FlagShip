import numpy as np
import multiprocessing as mp
from astropy.io import fits

from perfiles import lenscat_load

a = {
    'NCORES':10,
    'RMIN':0.0, 'RMAX':5.0, 'NBINS':50,
    'Rv_min':10.0, 'Rv_max':12.0, 'z_min':0.2, 'z_max':0.3, 'rho1_min':-1.0, 'rho1_max':-0.8, 'rho2_min':-1.0, 'rho2_max':100.0,
    'flag':2.0,
    'filename':'test', 'lensname':'server', 'tracname':'server',
}

if (a['tracname'] == 'local') or (a['lensname'] == 'local'):
    a['lensname'] = "/home/franco/FAMAF/Lensing/cats/MICE/voids_MICE.dat"
    a['tracname'] = "/home/franco/FAMAF/Lensing/cats/MICE/mice_halos_cut.fits"
else:
    a['tracname'] = '/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits'
    a['lensname'] = '/mnt/simulations/MICE/voids_MICE.dat'

a['filename'] = "radialprof_TEST.csv"

with fits.open(a['tracname']) as f:
    xhalo = f[1].data.xhalo
    yhalo = f[1].data.yhalo
    zhalo = f[1].data.zhalo
    lmhalo = f[1].data.lmhalo

mparticle = 2.93e10 # Msun/h
mask_particles = (lmhalo > np.log10(10*mparticle))
xhalo = xhalo[mask_particles]
yhalo = yhalo[mask_particles]
zhalo = zhalo[mask_particles]
lmhalo = lmhalo[mask_particles]

def get_halos(RMIN, RMAX,
              rv, xv, yv, zv):

    global xhalo, yhalo, zhalo, lmhalo
    distance = np.sqrt( (xhalo - xv)**2 + (yhalo - yv)**2 + (zhalo - zv)**2 ) / rv
    mask_ball = (distance < 5*RMAX) & (distance >= 0.0)
    mask_prof = (distance < RMAX) & (distance >= RMIN)

    massball = np.sum(10.0 ** lmhalo[mask_ball])
    halosball = len(lmhalo[mask_ball])

    logm = lmhalo[mask_prof]
    dist = distance[mask_prof]

    return logm, dist, massball, halosball

def partial_profile(RMIN,RMAX,NBINS,
                    rv, xv, yv, zv):
    
    NBINS = int(NBINS)
    logm, distance, massball, halosball = get_halos(RMIN, RMAX, rv, xv, yv, zv)

    NHalos = np.zeros(NBINS)
    mass = np.zeros(NBINS)

    DR = (RMAX-RMIN)/NBINS
    for lm,d in zip(logm,distance):
        ibin = np.floor((d-RMIN)/DR).astype(int)
        NHalos[ibin] += 1.0
        mass[ibin] += 10.0**lm

    return mass, NHalos, massball, halosball

L, _, _ = lenscat_load(a['Rv_min'], a['Rv_max'], a['z_min'], a['z_max'], a['rho1_min'], a['rho1_max'], a['rho2_min'], a['rho2_max'],
                        flag=a['flag'], lensname=a['lensname'],
                        split=False)
N = 10
for i in range(N):
    profs = partial_profile(a['RMIN'], a['RMAX'], a['NBINS'], L[1,i], L[5,i], L[6,i], L[7,i])
    name = f"void_py_{int(L[0,i])}.csv"
    np.savetxt(name, 
               np.column_stack(
                   [profs[0],profs[1],np.full(a['NBINS'],profs[2]),np.full(a['NBINS'],profs[3])]
                   ), 
                delimiter=','
            )
    