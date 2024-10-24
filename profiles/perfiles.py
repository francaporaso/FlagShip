import numpy as np
import multiprocessing as mp
from astropy.io import fits
from tqdm import tqdm

def lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, 
                 flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat",
                 split=False, NCORES=1):

    ## 0:id, 1:Rv, 2:ra, 3:dec, 4:z, 5:xv, 6:yv, 7:zv, 8:rho1, 9:rho2, 10:logp, 11:flag
    L = np.loadtxt(lensname).T

    mask = (L[1] >= Rv_min) & (L[1] <= Rv_max) & (L[4] >= z_min) & (L[4] <= z_max) & (
            L[8] >= rho1_min) & (L[8] <= rho1_max) & (L[9] >= rho2_min) & (L[9] <= rho2_max) & (L[11] >= flag)

    nvoids = mask.sum()
    L = L[:,mask]

    if split:
        lbins = int(round(nvoids/float(NCORES), 0))
        slices = ((np.arange(lbins)+1)*NCORES).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T,slices)

    return L, nvoids

def get_halos(RMIN, RMAX,
              rv, xv, yv, zv,
              tracname="/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits"):

    mp = 2.93e10 # Msun/h

    with fits.open(tracname) as f:
        xhalo = f[1].data.xhalo
        yhalo = f[1].data.yhalo
        zhalo = f[1].data.zhalo
        lmhalo = f[1].data.lmhalo

    distance = np.sqrt( (xhalo - xv)**2 + (yhalo - yv)**2 + (zhalo - zv)**2) / rv
    mask = (distance <= RMAX) & (distance >= RMIN) & (lmhalo > np.log10(10*mp))

    lmhalo = lmhalo[mask]
    distance = distance[mask]

    massball = np.sum(10.0 ** lmhalo)
    halosball = len(distance)

    return lmhalo, distance, massball, halosball

def partial_profile(RMIN,RMAX,NBINS,
                    rv, xv, yv, zv,
                    tracname=#"/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits"):
                    "/home/franco/FAMAF/Lensing/cats/MICE/mice_halos_cut.fits"):
    
    NBINS = int(NBINS)
    logm, distance, massball, halosball = get_halos(0.0, 5*RMAX, rv, xv, yv, zv, tracname=tracname)

    mask = (distance <= RMAX) & (distance >= RMIN)
    logm, distance = logm[mask], distance[mask]

    NHalos = np.zeros(NBINS)
    mass = np.zeros(NBINS)

    DR = (RMAX-RMIN)/NBINS
    for lm,d in zip(logm,distance):
        ibin = np.floor((d-RMIN)/DR).astype(int)
        NHalos[ibin] += 1.0
        mass[ibin] += 10.0**lm

    return mass, NHalos, massball, halosball

def partial_profile_unpack(myinput):
    return partial_profile(*myinput)

def stacking(NCORES, 
             RMIN, RMAX, NBINS,
             Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
             flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat",
             filename="pru_stack.csv"):
    
    L, nvoids = lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
                     flag=flag, lensname=lensname,
                     split=True, NCORES=NCORES)

    print(f"NVOIDS: .... {nvoids}")

    if nvoids < NCORES:
            NCORES = nvoids

    mass  = np.zeros(NBINS)
    halos = np.zeros(NBINS)
    massball = 0.0
    halosball = 0.0

    ### tqdm is progressbar
    ### TODO
    #### es probable que no sea necesario dividir L, simplemente usando ´chuncksize´ de Pool.map
    for i,Li in enumerate(tqdm(L)):

        num = len(Li)
        
        if num==1:
            entrada = np.array([
                RMIN, RMAX, NBINS,
                Li[1], Li[5], Li[6], Li[7],
            ])
            
            resmap = partial_profile(*entrada)

        else:
            RMIN_a = np.full(num, RMIN)
            RMAX_a = np.full(num, RMAX)
            NBINS_a = np.full(num, NBINS)
            entrada = np.array([
                RMIN_a, RMAX_a, NBINS_a, 
                Li.T[2], Li.T[5], Li.T[6], Li.T[7],
            ]).T
         
            with mp.Pool(processes=num) as pool:
                resmap = np.array(pool.map(partial_profile_unpack, entrada), dtype=object)
                pool.close()
                pool.join()

        for res in resmap:
            mass  += res[0]
            halos += res[1]
            massball  += res[2]
            halosball += res[3]

    
    meandenball   = massball/(4*np.pi/3 * (5*RMAX)**3)
    meanhalosball = halosball/(4*np.pi/3 * (5*RMAX)**3)

    DR = (RMAX-RMIN)/NBINS
    
    vol = np.zeros(NBINS)
    volcum = np.zeros(NBINS)
    for k in range(NBINS):
        vol[k]    = ((k+1.0)*DR + RMIN)**3 - (k*DR + RMIN)**3
        volcum[k] = ((k+1.0)*DR + RMIN)**3
    
    vol *= (4*np.pi/3)
    volcum *= (4*np.pi/3)

    Delta    = (mass/vol)/meandenball - 1
    DeltaCum = (np.cumsum(mass)/volcum)/meandenball - 1
    DeltaHalos    = (halos/vol)/meanhalosball - 1
    DeltaHalosCum = (np.cumsum(halos)/volcum)/meanhalosball - 1

    print(f"Saving in: {filename}")

    # Stack the arrays column-wise and save
    data = np.column_stack((Delta, DeltaCum, DeltaHalos, DeltaHalosCum))
    np.savetxt(filename, data, delimiter=',')

    print("END!")


if __name__ == "__main___":

    NCORES = 4
    RMIN, RMAX, NBINS = 0.0, 5.0, 50
    Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, flag = 10.0, 12.0, 0.2, 0.25, -1.0, -0.8, -1.0, 100.0, 2.0
    filename = "radialprof_stack_R_{:.0f}_{:.0f}_z{:.1f}_{:.1f}.csv".format(Rv_min, Rv_max, z_min, z_max)
    lensname = "/home/franco/FAMAF/Lensing/cats/MICE/voids_MICE.dat"
    tracname = "/home/franco/FAMAF/Lensing/cats/MICE/mice_halos_cut.fits"
    # lensname = "/mnt/simulations/MICE/voids_MICE.dat"
    # tracname = "/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits"

    stacking(
        NCORES, 
        RMIN, RMAX, NBINS,
        Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
        flag=flag, lensname=lensname,
        filename=filename,
    )
