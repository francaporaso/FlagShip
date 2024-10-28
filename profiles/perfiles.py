import numpy as np
import multiprocessing as mp
from astropy.io import fits
from tqdm import tqdm

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
        lbins = int(round(nvoids/float(NSPLITS), 0))
        slices = ((np.arange(lbins)+1)*NSPLITS).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

## TODO
## cambiar por np.where ? capaz hay problemas con la mascara con python nativo....
def get_halos(RMIN, RMAX,
              rv, xv, yv, zv):
    
    ###  TEST debugging
    print(xhalo.shape)

    distance = np.sqrt( (xhalo - xv)**2 + (yhalo - yv)**2 + (zhalo - zv)**2 ) / rv
    mask_ball = (distance < 5*RMAX) & (distance >= 0.0)
    mask_prof = (distance < RMAX) & (distance >= RMIN)

    massball = np.sum(10.0 ** lmhalo[mask_ball])
    halosball = len(lmhalo[mask_ball])

    return lmhalo[mask_prof], distance[mask_prof], massball, halosball

## TODO
## agregar solucion a edgecases cuando el perfil se escapa de la caja
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

def partial_profile_unpack(myinput):
    return partial_profile(*myinput)

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

    if nvoids < NCORES:
            NCORES = nvoids

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
                Li[1], Li[5], Li[6], Li[7],
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
                j += 1
                mass  += np.tile(res[0], (nk+1,1))*km
                halos += np.tile(res[1], (nk+1,1))*km
                massball  += (np.tile(res[2], (nk+1,1))*km)[:,0]
                halosball += (np.tile(res[3], (nk+1,1))*km)[:,0]
                
                ### TODO
                ## chequear que no haya competencia acá...
                ## es mas seguro con la forma de forVoid_slice_MICE.py
                # for res in pool.imap(partial_profile_unpack, entrada):
                #     km = np.tile(K[i][j], (NBINS,1)).T
                #     j += 1
                #     mass  += np.tile(res[0], (nk+1,1))*km
                #     halos += np.tile(res[1], (nk+1,1))*km
                #     massball  += (np.tile(res[2], (nk+1,1))*km)[:,0]
                #     halosball += (np.tile(res[3], (nk+1,1))*km)[:,0]


    meandenball   = (massball/(4*np.pi/3 * (5*RMAX)**3))
    meanhalosball = (halosball/(4*np.pi/3 * (5*RMAX)**3))

    DR = (RMAX-RMIN)/NBINS
    
    vol    = np.zeros(NBINS)
    volcum = np.zeros(NBINS)
    for k in range(NBINS):
        vol[k]    = ((k+1.0)*DR + RMIN)**3 - (k*DR + RMIN)**3
        volcum[k] = ((k+1.0)*DR + RMIN)**3
    
    vol    *= (4*np.pi/3)
    volcum *= (4*np.pi/3)

    Delta    = np.zeros((nk+1, NBINS))
    DeltaCum = np.zeros((nk+1, NBINS))
    DeltaHalos    = np.zeros((nk+1, NBINS))
    DeltaHalosCum = np.zeros((nk+1, NBINS))

    for i in range(nk+1):
        Delta[i,:]    = (mass[i]/vol)/meandenball[i] - 1
        DeltaCum[i,:] = (np.cumsum(mass[i])/volcum)/meandenball[i] - 1
        DeltaHalos[i,:]    = (halos[i]/vol)/meanhalosball[i] - 1
        DeltaHalosCum[i,:] = (np.cumsum(halos[i])/volcum)/meanhalosball[i] - 1

    ## calculating covariance matrix
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
    data = np.column_stack((Delta[0], DeltaCum[0], DeltaHalos[0], DeltaHalosCum[0]))
    np.savetxt(filename, data, delimiter=',')
    np.savetxt('cov_delta'+filename, cov_delta, delimiter=',')
    np.savetxt('cov_deltacum'+filename, cov_deltacum, delimiter=',')
    np.savetxt('cov_deltahalos'+filename, cov_deltahalos, delimiter=',')
    np.savetxt('cov_deltahaloscum'+filename, cov_deltahaloscum, delimiter=',')

    print("END!")

    return 0
    ### ---------------------------------------------------------------TEST!
    
    # print('---------------------------------TEST!')

    # np.savetxt('test_masa.csv', mass, delimiter=',')
    # np.savetxt('test_halos.csv', halos, delimiter=',')
    # np.savetxt('test_ball.csv', np.column_stack([massball, halosball]), delimiter=',')

    # print("END!")

if __name__ == "__main__":

    ### TODO 
    ### el perfil parece estar yendo a 0.2 en vez de 0.... chequear......

    import argparse as ag
    
    options = {
        '--NCORES':100,
        '--RMIN':0.0, '--RMAX':5.0, '--NBINS':50,
        '--Rv_min':10.0, '--Rv_max':12.0, '--z_min':0.2, '--z_max':0.3, '--rho1_min':-1.0, '--rho1_max':-0.8, '--rho2_min':-1.0, '--rho2_max':100.0,
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

    # if (a.filename!='test') or (a.filename!='pru'):
    a.filename = "radialprof_R{:.0f}_{:.0f}_z{:.1f}_{:.1f}_type{}_v2.csv".format(a.Rv_min, a.Rv_max, a.z_min, a.z_max, tipo)

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

    stacking(
        a.NCORES, 
        a.RMIN, a.RMAX, a.NBINS,
        a.Rv_min, a.Rv_max, a.z_min, a.z_max, a.rho1_min, a.rho1_max, a.rho2_min, a.rho2_max,
        flag=a.flag, lensname=a.lensname,
        filename=a.filename,
    )

    # NCORES = 100
    # RMIN, RMAX, NBINS = 0.0, 5.0, 50
    # Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, flag = 10.0, 12.0, 0.2, 0.3, -1.0, -0.8, -1.0, 100.0, 2.0
    # # filename = "radialprof_stack_R_{:.0f}_{:.0f}_z{:.1f}_{:.1f}_2.csv".format(Rv_min, Rv_max, z_min, z_max)
    # filename = "radialprof_stack_TEST.csv"
    # # lensname = "/home/franco/FAMAF/Lensing/cats/MICE/voids_MICE.dat"
    # # tracname = "/home/franco/FAMAF/Lensing/cats/MICE/mice_halos_cut.fits"
    # lensname = "/mnt/simulations/MICE/voids_MICE.dat"
    # tracname = "/home/fcaporaso/cats/MICE/mice_halos_centralesF.fits"
