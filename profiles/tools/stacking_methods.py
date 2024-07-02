import numpy as np
from astropy.io import fits
import time
from multiprocessing import Pool

import sys

from tools.void import Void, Tracer

def method1(tcat,
            xv,yv,zv,rv,
            RMIN,RMAX,dr):
    '''
    making void objects all at once and storing in voids_list
    very similar to method2
    speeds up when calling get_tracers here
    '''

    voids_list = list(map(Void, xv,yv,zv,rv))
    tot_tracers = 0
    for v in voids_list:
        v.get_tracers(cat=tcat, RMAX=RMAX+dr/2, center=False)
        tot_tracers += len(v.tr)
    print(f'N tracers: {tot_tracers}')
    
    delta_list = np.array([v.radial_density_profile(cat=tcat, RMIN=RMIN, RMAX=RMAX, dr=dr) for v in voids_list])


    stacked_profile = np.mean(delta_list, axis=0)
    return stacked_profile

def method2(tcat,
            xv,yv,zv,rv,
            RMIN,RMAX,dr):
    '''
    making void objects one by one and overwriting v
    SLOWEST METHOD
    speeds up when calling get_tracers here
    '''

    NBINS = int(round(((RMAX-RMIN)/dr),0))
    nvoids = len(xv)

    stacked_profile = np.zeros(NBINS)
    tot_tracers = 0
    
    for i in range(nvoids):
        v = Void(xv[i], yv[i], zv[i], rv[i])
        v.get_tracers(cat=tcat, RMAX=RMAX+dr/2, center=False)
        tot_tracers += len(v.tr)
        v.sort_tracers()
        stacked_profile += v.radial_density_profile(cat=tcat, RMIN=RMIN, RMAX=RMAX, dr=dr)

    print(f'N tracers: {tot_tracers}')

    stacked_profile /= nvoids
    return stacked_profile
    
def method3(xv:list[float], yv:list[float], zv:list[float], rv:list[float],
            RMIN:float, RMAX:float, dr:float):
    '''
    creating a void object with all tracers from the individual voids
    FASTEST METHOD (twice as met1 and 2)
    '''

    Nvoids = len(xv)

    tr_list = []

    for i in range(Nvoids):
        v = Void(xv[i], yv[i], zv[i], rv[i])
        v.get_tracers(RMAX=RMAX+dr/10, center=True)
        tr_list += v.tr

    # return tr_list

    ### usar cuando no hay paralelizado

    print(f'N tracers: {len(tr_list)}')

    stacked_void = Void(0.,0.,0.,1.)
    stacked_void.tr = tr_list  
    stacked_void.sort_tracers()

    stacked_profile = stacked_void.radial_density_profile(RMIN=RMIN, RMAX=RMAX, dr=dr)

    return stacked_profile/Nvoids

def method3_singlevoid(xv,yv,zv,rv,
                       RMIN,RMAX,dr):

    tr_list = []

    v = Void(xv, yv, zv, rv)
    v.get_tracers(RMAX=RMAX+dr/10, center=True)
    tr_list += v.tr

    return tr_list

def method3_unpack(args):
    return method3_singlevoid(*args)

def parallel3(ncores:int,
              L:list[float],
              RMIN:float, RMAX:float, dr:float):

    #split voids cat
    Nvoids = len(L[1])
    if Nvoids < ncores:
        ncores = Nvoids

    lbins = int(round(Nvoids/float(ncores), 0))
    slices = ((np.arange(lbins)+1)*ncores).astype(int)
    slices = slices[(slices < Nvoids)]
    Lsplit = np.split(L.T,slices)

    del L

    #for split

    LARGO = len(Lsplit)
    timeslice = np.array([])
    all_tracers = []
    
    print(f'Running in parallel with {ncores} cores')

    for l, L_l in enumerate(Lsplit):
        print(f'RUN {l+1} OF {LARGO}')            
        t1 = time.time()
        lensnum = len(L_l)
            
        if lensnum == 1:
            # tcat,
            # xv,yv,zv,rv,
            # RMIN,RMAX,dr
            entrada = [
                L_l.T[5],L_l.T[6],L_l.T[7],L_l.T[1],
                RMIN, RMAX, dr,
            ]
            
            salida = [method3_unpack(entrada)]

        else:                
            rmin_arr = np.full(lensnum, RMIN)
            rmax_arr = np.full(lensnum, RMAX)
            dr_arr   = np.full(lensnum, dr)

            entrada = np.array([
                L_l.T[5], L_l.T[6], L_l.T[7], L_l.T[1],
                rmin_arr, rmax_arr, dr_arr,
            ]).T

            with Pool(processes=lensnum) as pool:
                salida = np.array(pool.map(method3_unpack,entrada))
                pool.close()
                pool.join()

        #join parts

        for tr_list in salida:
            all_tracers += tr_list

        t2 = time.time()
        ts = (t2-t1)/60.
        timeslice = np.append(timeslice, ts)
        print('TIME SLICE')
        print(f'{np.round(ts,4)} min')
        print('Estimated remaining time')
        print(f'{np.round(np.mean(timeslice)*(LARGO-(l+1)), 3)} min')

    # calculating stacked profile
    print('Paralelization ended')
    print('--------')

    print(f'N tracers: {len(all_tracers)}')

    stacked_void = Void(0.,0.,0.,1.)
    stacked_void.tr = all_tracers
    stacked_void.sort_tracers()

    void_profile = stacked_void.radial_density_profile(RMIN=RMIN, RMAX=RMAX, dr=dr)
    void_profile /= Nvoids

    return void_profile
