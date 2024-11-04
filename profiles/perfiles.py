import numpy as np
import multiprocessing as mp
from astropy.io import fits
from tqdm import tqdm
# from numba import njit, jit

def cov_matrix(array):
        
    K = len(array)
    Kmean = np.average(array,axis=0)
    bins = array.shape[1]
    
    COV = np.zeros((bins,bins))
    
    for k in range(K):
        dif = (array[k]- Kmean)
        COV += np.outer(dif,dif)        
    
    COV *= (K-1)/K
    return COV

def lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, 
                 flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat",
                 split=False, NSPLITS=1):

    ## 0:id, 1:Rv, 2:ra, 3:dec, 4:z, 5:xv, 6:yv, 7:zv, 8:rho1, 9:rho2, 10:logp, 11:flag
    L = np.loadtxt(lensname).T

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

    mask = (L[1] >= Rv_min) & (L[1] < Rv_max) & (L[4] >= z_min) & (L[4] < z_max) & (
            L[8] >= rho1_min) & (L[8] < rho1_max) & (L[9] >= rho2_min) & (L[9] < rho2_max) & (L[11] >= flag)

    nvoids = mask.sum()
    L = L[:,mask]

    if split:
        if nvoids < NSPLITS:
            NSPLITS = nvoids
        lbins = int(round(nvoids/float(NSPLITS), 0))
        slices = ((np.arange(lbins)+1)*NSPLITS).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

# @njit
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

## TODO
## agregar solucion a edgecases cuando el perfil se escapa de la caja
def partial_profile(RMIN,RMAX,NBINS,
                    rv, xv, yv, zv, vid):
    
    NBINS = int(NBINS)
    logm, distance, massball, halosball = get_halos(RMIN, RMAX, rv, xv, yv, zv)

    NHalos = np.zeros(NBINS)
    mass = np.zeros(NBINS)

    DR = (RMAX-RMIN)/NBINS
    for lm,d in zip(logm,distance):
        ibin = int(np.floor((d-RMIN)/DR))
        NHalos[ibin] += 1.0
        mass[ibin] += 10.0**lm

    return mass, NHalos, massball, halosball, np.full(NBINS, vid)

def partial_profile2(RMIN,RMAX,NBINS,
                    rv, xv, yv, zv, 
                    vid):
    
    NBINS = int(NBINS)
    logm, distance, massball, halosball = get_halos(RMIN, RMAX, rv, xv, yv, zv)

    NHalos = np.zeros(NBINS)
    mass = np.zeros(NBINS)

    DR = (RMAX-RMIN)/NBINS
    for lm,d in zip(logm,distance):
        ibin = int(np.floor((d-RMIN)/DR))
        NHalos[ibin] += 1.0
        mass[ibin] += 10.0**lm

    vol = np.array([((k+1.0)*DR + RMIN)**3 - (k*DR + RMIN)**3 for k in range(NBINS)]) * rv**3
        
    return mass, NHalos, massball, halosball, vol, np.full(NBINS, vid)

def partial_profile_unpack(myinput):
    return partial_profile(*myinput)

def partial_profile_unpack2(myinput):
    return partial_profile2(*myinput)

def individual_profile(RMIN, RMAX, NBINS,
                       rv, xv, yv, zv, vid):
    
    NBINS = int(NBINS)
    mass, halos, massball, halosball = partial_profile(RMIN, RMAX, NBINS, rv, xv, yv, zv)
    
    meandenball = massball/(4/3*np.pi * (5*RMAX*rv)**3)
    meanhalosball = halosball/(4/3*np.pi * (5*RMAX*rv)**3)

    DR = (RMAX-RMIN)/NBINS
    
    vol    = np.zeros(NBINS)
    volcum = np.zeros(NBINS)
    for k in range(NBINS):
        vol[k]    = ((k+1.0)*DR + RMIN)**3 - (k*DR + RMIN)**3
        volcum[k] = ((k+1.0)*DR + RMIN)**3

    vol    *= 4/3*np.pi * rv**3
    volcum *= 4/3*np.pi * rv**3

    Delta    = (mass/vol)/meandenball - 1
    DeltaCum = (np.cumsum(mass)/volcum)/meandenball - 1
    DeltaHalos    = (halos/vol)/meanhalosball - 1
    DeltaHalosCum = (np.cumsum(halos)/volcum)/meanhalosball - 1

    return Delta, DeltaCum, DeltaHalos, DeltaHalosCum, np.full(NBINS, vid)

def individual_profile_unpack(myinput):
    return individual_profile(*myinput)


## TODO
## cambiar guardado en .csv por .fits
def averaging(NCORES, 
              RMIN, RMAX, NBINS,
              Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
              flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat", filename="pru_stack.csv"):

    nk = 100
    L, K, nvoids = lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
                                flag=flag, lensname=lensname,
                                split=True, NSPLITS=NCORES)

    print(f"NVOIDS: .... {nvoids}")

    Delta    = np.zeros((nvoids, nk+1,NBINS))
    DeltaCum = np.zeros((nvoids, nk+1,NBINS))
    DeltaHalos    = np.zeros((nvoids, nk+1,NBINS))
    DeltaHalosCum = np.zeros((nvoids, nk+1,NBINS))

    ### TODO
    #### es probable que no sea necesario dividir L, simplemente usando ´chuncksize´ de Pool.map
    for i,Li in enumerate(tqdm(L)):

        num = len(Li)
        if num==1:
            entrada = np.array([
                RMIN, RMAX, NBINS,
                Li[0][1], Li[0][5], Li[0][6], Li[0][7],
            ])
            
            resmap = individual_profile(*entrada)

        else:
            RMIN_a = np.full(num, RMIN)
            RMAX_a = np.full(num, RMAX)
            NBINS_a = np.full(num, NBINS)
            entrada = np.array([
                RMIN_a, RMAX_a, NBINS_a, 
                Li.T[1], Li.T[5], Li.T[6], Li.T[7],
            ]).T

            with mp.Pool(processes=num) as pool:
                resmap = pool.map(individual_profile_unpack, entrada)
                pool.close()
                pool.join()
            
        for j,res in enumerate(resmap):
            km = np.tile(K[i][j], (NBINS,1)).T
            Delta[i*num+j]    = np.tile(res[0], (nk+1,1))*km
            DeltaCum[i*num+j] = np.tile(res[1], (nk+1,1))*km
            DeltaHalos[i*num+j]    = np.tile(res[2], (nk+1,1))*km
            DeltaHalosCum[i*num+j] = np.tile(res[3], (nk+1,1))*km

    ## TODO
    ## guardar los individuales... 
    Delta_m    = np.mean(Delta, axis=0)
    DeltaCum_m = np.mean(DeltaCum, axis=0)
    DeltaHalos_m    = np.mean(DeltaHalos, axis=0)
    DeltaHalosCum_m = np.mean(DeltaHalosCum, axis=0)

    # calculating covariance matrix
    cov_delta    = cov_matrix(Delta_m[1:,:])
    cov_deltacum = cov_matrix(DeltaCum_m[1:,:])
    cov_deltahalos    = cov_matrix(DeltaHalos_m[1:,:])
    cov_deltahaloscum = cov_matrix(DeltaHalosCum_m[1:,:])

    folder = 'profiles/results/'
    print(f"Saving in: {filename}")
    print(f"Saving in: {'cov_delta'+filename}")
    print(f"Saving in: {'cov_deltacum'+filename}")
    print(f"Saving in: {'cov_deltahalos'+filename}")
    print(f"Saving in: {'cov_deltahaloscum'+filename}")

    # Stack the arrays column-wise and save
    data = np.column_stack((Delta_m[0], DeltaCum_m[0], DeltaHalos_m[0], DeltaHalosCum_m[0]))
    np.savetxt(folder+filename, data, delimiter=',')
    np.savetxt(folder+'cov_delta'+filename, cov_delta, delimiter=',')
    np.savetxt(folder+'cov_deltacum'+filename, cov_deltacum, delimiter=',')
    np.savetxt(folder+'cov_deltahalos'+filename, cov_deltahalos, delimiter=',')
    np.savetxt(folder+'cov_deltahaloscum'+filename, cov_deltahaloscum, delimiter=',')

    print("END!")

    return 0

## TODO
## cambiar guardado en .csv por .fits
def stacking(NCORES, 
             RMIN, RMAX, NBINS,
             Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
             flag=2.0, lensname="/mnt/simulations/MICE/voids_MICE.dat", filename="pru_stack.csv"):
    
    nk = 100
    L, K, nvoids = lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
                                flag=flag, lensname=lensname,
                                split=True, NSPLITS=NCORES)

    print(f"NVOIDS: .... {nvoids}")

    mass  = np.zeros((nk+1,NBINS))
    halos = np.zeros((nk+1,NBINS))
    massball  = np.zeros(nk+1)
    halosball = np.zeros(nk+1)

    ### TODO
    #### es probable que no sea necesario dividir L, simplemente usando ´chuncksize´ de Pool.map
    for i,Li in enumerate(tqdm(L)):

        num = len(Li)
        if num==1:
            entrada = np.array([
                RMIN, RMAX, NBINS,
                Li[0][1], Li[0][5], Li[0][6], Li[0][7],
            ])
            
            resmap = partial_profile(*entrada)

        else:
            RMIN_a = np.full(num, RMIN)
            RMAX_a = np.full(num, RMAX)
            NBINS_a = np.full(num, NBINS)
            entrada = np.array([
                RMIN_a, RMAX_a, NBINS_a, 
                Li.T[1], Li.T[5], Li.T[6], Li.T[7],
            ]).T

            with mp.Pool(processes=num) as pool:
                resmap = pool.map(partial_profile_unpack, entrada)
                pool.close()
                pool.join()
            
        j = 0
        for res in resmap:
            km = np.tile(K[i][j], (NBINS,1)).T
            
            mass  += np.tile(res[0], (nk+1,1))*km
            halos += np.tile(res[1], (nk+1,1))*km
            massball  += (np.tile(res[2], (nk+1,1))*km)[:,0]
            halosball += (np.tile(res[3], (nk+1,1))*km)[:,0]
            j += 1

    meandenball = massball/(4/3*np.pi * (5*RMAX)**3)
    meanhalosball = halosball/(4/3*np.pi * (5*RMAX)**3)

    DR = (RMAX-RMIN)/NBINS
    
    vol    = np.zeros(NBINS)
    volcum = np.zeros(NBINS)
    for k in range(NBINS):
        vol[k]    = ((k+1.0)*DR + RMIN)**3 - (k*DR + RMIN)**3
        volcum[k] = ((k+1.0)*DR + RMIN)**3
    
    ### Volumen incorrecto... está asignando el vol del bin k y dsp lo divide con la masa del bin k-1
    ### sin embargo, así da un perfil decente....
    # for k in range(1,NBINS+1):
    #     vol[k-1]    = ((k+1)*DR + RMIN)**3 - (k*DR + RMIN)**3
    #     volcum[k-1] = ((k+1)*DR + RMIN)**3

    vol    *= (4*np.pi/3)
    volcum *= (4*np.pi/3)

    Delta    = np.zeros((nk+1, NBINS))
    DeltaCum = np.zeros((nk+1, NBINS))
    DeltaHalos    = np.zeros((nk+1, NBINS))
    DeltaHalosCum = np.zeros((nk+1, NBINS))

    for i in range(nk+1):
        Delta[i]    = (mass[i]/vol)/meandenball[i] - 1
        DeltaCum[i] = (np.cumsum(mass[i])/volcum)/meandenball[i] - 1
        DeltaHalos[i]    = (halos[i]/vol)/meanhalosball[i] - 1
        DeltaHalosCum[i] = (np.cumsum(halos[i])/volcum)/meanhalosball[i] - 1

    # calculating covariance matrix
    cov_delta    = cov_matrix(Delta[1:,:])
    cov_deltacum = cov_matrix(DeltaCum[1:,:])
    cov_deltahalos    = cov_matrix(DeltaHalos[1:,:])
    cov_deltahaloscum = cov_matrix(DeltaHalosCum[1:,:])

    print(f"Saving in: {filename}")
    print(f"Saving in: {'cov_delta'+filename}")
    print(f"Saving in: {'cov_deltacum'+filename}")
    print(f"Saving in: {'cov_deltahalos'+filename}")
    print(f"Saving in: {'cov_deltahaloscum'+filename}")

    # Stack the arrays column-wise and save
    # data = np.column_stack((Delta[0], DeltaCum[0], DeltaHalos[0], DeltaHalosCum[0]))
    data = np.column_stack((Delta, DeltaCum, DeltaHalos, DeltaHalosCum))
    np.savetxt(filename, data, delimiter=',')
    np.savetxt('cov_delta'+filename, cov_delta, delimiter=',')
    np.savetxt('cov_deltacum'+filename, cov_deltacum, delimiter=',')
    np.savetxt('cov_deltahalos'+filename, cov_deltahalos, delimiter=',')
    np.savetxt('cov_deltahaloscum'+filename, cov_deltahaloscum, delimiter=',')

    print("END!")

    return 0

def all_individuals(NCORES, 
                    RMIN,RMAX, NBINS, 
                    Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max):
    
    print("Ejecutando all_individuals para perfiles individuales de masa")

    L,_,nvoids = lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
                              split=True, NSPLITS=NCORES)
    
    print(f"NVOIDS: {nvoids}")
    Delta  = np.zeros((nvoids, NBINS))
    DeltaCum = np.zeros((nvoids, NBINS))
    DeltaHalos = np.zeros((nvoids, NBINS))
    DeltaHalosCum = np.zeros((nvoids, NBINS))
    VoidID = np.zeros((nvoids, NBINS))

    for i,Li in enumerate(tqdm(L)):

        num = len(Li)
        if num==1:
            entrada = np.array([
                RMIN, RMAX, NBINS,
                Li[0][1], Li[0][5], Li[0][6], Li[0][7], Li[0][0],
            ])
            
            resmap = partial_profile2(*entrada)

        else:
            RMIN_a = np.full(num, RMIN)
            RMAX_a = np.full(num, RMAX)
            NBINS_a = np.full(num, NBINS)
            entrada = np.array([
                RMIN_a, RMAX_a, NBINS_a, 
                Li.T[1], Li.T[5], Li.T[6], Li.T[7], Li.T[0],
            ]).T

            with mp.Pool(processes=num) as pool:
                resmap = pool.map(partial_profile_unpack2, entrada)
                pool.close()
                pool.join()
            
        for j,res in enumerate(resmap):
            np.savetxt(
                f'profiles/results/mass/void_{int(res[5][0])}.csv',
                np.column_stack(
                    [res[0],res[1],res[2],res[3],res[4]]
                ), delimiter=','
            )

    print('END!')

if __name__ == "__main__":

    ### TODO 
    ### el perfil parece estar yendo a 0.2 en vez de 0.... chequear......

    import argparse as ag
    
    options = {
        '--NCORES':32,
        '--RMIN':0.0, '--RMAX':5.0, '--NBINS':200,
        '--Rv_min':10.0, '--Rv_max':11.0, '--z_min':0.2, '--z_max':0.21, '--rho1_min':-1.0, '--rho1_max':-0.8, '--rho2_min':-1.0, '--rho2_max':100.0,
        '--flag':2.0,
        '--filename':'test', '--lensname':'server', '--tracname':'server',
    }
    
    parser = ag.ArgumentParser()
    for key,value in options.items():
        if key[-4:]=='name':
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=str)
        elif (key[2:]=='NBINS') or (key[2:]=='NCORES'):
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=int)
        else:
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=float)

    a = parser.parse_args()

    if (a.tracname == 'local') or (a.lensname == 'local'):
        a.lensname = "/home/franco/FAMAF/Lensing/cats/MICE/voids_MICE.dat"
        a.tracname = "/home/franco/FAMAF/Lensing/cats/MICE/mice_halos_cut.fits"
    else:
        a.tracname = '/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits'
        a.lensname = '/mnt/simulations/MICE/voids_MICE.dat'

    if a.rho2_min <= 0.0 and a.rho2_max <= 0.0 :
        tipo = 'R'
    elif a.rho2_min >= 0.0 and a.rho2_max >= 0.0:
        tipo = 'S'
    else:
        tipo = 'A'

    if (a.filename[:4] !='test'):
        a.filename = "averageradialprof_R{:.0f}_{:.0f}_z{:.1f}_{:.1f}_type{}".format(a.Rv_min, a.Rv_max, a.z_min, a.z_max, tipo)
    a.filename += '.csv'

    ## opening tracers file and general masking
    with fits.open(a.tracname) as f:
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

    all_individuals(
        a.NCORES,
        a.RMIN, a.RMAX, a.NBINS,
        a.Rv_min, a.Rv_max, a.z_min, a.z_max, a.rho1_min, a.rho1_max, a.rho2_min, a.rho2_max
    )

    # averaging(
    #     a.NCORES, 
    #     a.RMIN, a.RMAX, a.NBINS,
    #     a.Rv_min, a.Rv_max, a.z_min, a.z_max, a.rho1_min, a.rho1_max, a.rho2_min, a.rho2_max,
    #     flag=a.flag, lensname=a.lensname,
    #     filename=a.filename,
    # )

    # stacking(
    #     a.NCORES, 
    #     a.RMIN, a.RMAX, a.NBINS,
    #     a.Rv_min, a.Rv_max, a.z_min, a.z_max, a.rho1_min, a.rho1_max, a.rho2_min, a.rho2_max,
    #     flag=a.flag, lensname=a.lensname,
    #     filename=a.filename,
    # )

    # NCORES = 100
    # RMIN, RMAX, NBINS = 0.0, 5.0, 50
    # Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, flag = 10.0, 12.0, 0.2, 0.3, -1.0, -0.8, -1.0, 100.0, 2.0
    # # filename = "radialprof_stack_R_{:.0f}_{:.0f}_z{:.1f}_{:.1f}_2.csv".format(Rv_min, Rv_max, z_min, z_max)
    # filename = "radialprof_stack_TEST.csv"
    # # lensname = "/home/franco/FAMAF/Lensing/cats/MICE/voids_MICE.dat"
    # # tracname = "/home/franco/FAMAF/Lensing/cats/MICE/mice_halos_cut.fits"
    # lensname = "/mnt/simulations/MICE/voids_MICE.dat"
    # tracname = "/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits"
