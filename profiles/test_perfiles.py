import numpy as np
import multiprocessing as mp
from astropy.io import fits

from perfiles import lenscat_load, get_halos, partial_profile, partial_profile_unpack

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

def perfiles_serie():

    L, _, _ = lenscat_load(a['Rv_min'], a['Rv_max'], a['z_min'], a['z_max'], a['rho1_min'], a['rho1_max'], a['rho2_min'], a['rho2_max'],
                            flag=a['flag'], lensname=a['lensname'],
                            split=False)

    mass  = np.zeros(a['NBINS'])
    halos = np.zeros(a['NBINS'])
    massball  = 0.0
    halosball = 0.0

    profs = [partial_profile(a['RMIN'], a['RMAX'], a['NBINS'], L[1,0], L[5,0], L[6,0], L[7,0]) for i in range(50)]

    for res in profs:
        mass  += res[0]
        halos += res[1]
        massball  += res[2]
        halosball += res[3]

    meandenball   = (massball/(4*np.pi/3 * (5*a['RMAX'])**3))
    meanhalosball = (halosball/(4*np.pi/3 * (5*a['RMAX'])**3))

    DR = (a['RMAX']-a['RMIN'])/a['NBINS']    
    
    vol    = np.zeros(a['NBINS'])
    volcum = np.zeros(a['NBINS'])
    for k in range(a['NBINS']):
        vol[k]    = ((k+1.0)*DR + a['RMIN'])**3 - (k*DR + a['RMIN'])**3
        volcum[k] = ((k+1.0)*DR + a['RMIN'])**3
    
    vol    *= (4*np.pi/3)
    volcum *= (4*np.pi/3)

    Delta = mass/vol/meandenball - 1
    DeltaHalos = halos/vol/meanhalosball - 1
    DeltaCum = np.cumsum(mass)/volcum/meandenball - 1
    DeltaHalosCum = np.cumsum(halos)/volcum/meanhalosball - 1

    print(f"Saving in: 'serie_'{a['filename']}")
    data = np.column_stack((Delta, DeltaCum, DeltaHalos, DeltaHalosCum))
    np.savetxt("serie_"+a['filename'], data, delimiter=',')

    return 0

def perfiles_paralelo():
    
    L, _, _ = lenscat_load(a['Rv_min'], a['Rv_max'], a['z_min'], a['z_max'], a['rho1_min'], a['rho1_max'], a['rho2_min'], a['rho2_max'],
                                flag=a['flag'], lensname=a['lensname'],
                                split=True, NSPLITS=a['NCORES'])

    mass  = np.zeros(a['NBINS'])
    halos = np.zeros(a['NBINS'])
    massball  = 0.0
    halosball = 0.0

    ### TODO
    #### es probable que no sea necesario dividir L, simplemente usando ´chuncksize´ de Pool.map
    for i,Li in enumerate(L):

        num = len(Li)
        if num==1:
            entrada = np.array([
                a['NBINS'],
                Li[1], Li[5], Li[6], Li[7],
            ])
            
            resmap = partial_profile(*entrada)

        else:
            RMIN_a = np.full(num, a['RMIN'])
            RMAX_a = np.full(num, a['RMAX'])
            NBINS_a = np.full(num, a['NBINS'])
            entrada = np.array([
                RMIN_a, RMAX_a, NBINS_a, 
                Li.T[1], Li.T[5], Li.T[6], Li.T[7],
            ]).T

            with mp.Pool(processes=num) as pool:
                resmap = pool.map(partial_profile_unpack, entrada)
                pool.close()
                pool.join()
            
            for res in resmap:
                mass  += res[0]
                halos += res[1]
                massball  += res[2]
                halosball += res[3]

        if i==4:
            break ## así agarro 50 voids (i=4 => 5 it del for externo => 10 voids/it* 5 it = 50 voids)

    meandenball   = (massball/(4*np.pi/3 * (5*a['RMAX'])**3))
    meanhalosball = (halosball/(4*np.pi/3 * (5*a['RMAX'])**3))

    DR = (a['RMAX']-a['RMIN'])/a['NBINS']
    
    vol    = np.zeros(a['NBINS'])
    volcum = np.zeros(a['NBINS'])
    for k in range(a['NBINS']):
        vol[k]    = ((k+1.0)*DR + a['RMIN'])**3 - (k*DR + a['RMIN'])**3
        volcum[k] = ((k+1.0)*DR + a['RMIN'])**3
    
    vol    *= (4*np.pi/3)
    volcum *= (4*np.pi/3)

    Delta    = np.zeros(a['NBINS'])
    DeltaCum = np.zeros(a['NBINS'])
    DeltaHalos    = np.zeros(a['NBINS'])
    DeltaHalosCum = np.zeros(a['NBINS'])

    Delta = mass/vol/meandenball - 1
    DeltaHalos = halos/vol/meanhalosball - 1
    DeltaCum = np.cumsum(mass)/volcum/meandenball - 1
    DeltaHalosCum = np.cumsum(halos)/volcum/meanhalosball - 1    

    print(f"Saving in: 'paralelo_'{a['filename']}")
    data = np.column_stack((Delta, DeltaCum, DeltaHalos, DeltaHalosCum))
    np.savetxt("paralelo_"+a['filename'], data, delimiter=',')

    return 0

perfiles_serie()
perfiles_paralelo()
