import numpy as np
from astropy.io import fits

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
    
def method3(tcat,
            xv,yv,zv,rv,
            RMIN,RMAX,dr):
    '''
    creating a void object with all tracers from the individual voids
    FASTEST METHOD (twice as met1 and 2)
    '''

    nvoids = len(xv)

    tr_list = []

    for i in range(nvoids):
        v = Void(xv[i], yv[i], zv[i], rv[i])
        v.get_tracers(cat=tcat, RMAX=RMAX+dr/2, center=True)
        tr_list += v.tr

    print(f'N tracers: {len(tr_list)}')

    stacked_void = Void(0.,0.,0.,1.)
    stacked_void.tr = tr_list  
    stacked_void.sort_tracers()

    stacked_profile = stacked_void.radial_density_profile(cat=tcat, RMIN=RMIN, RMAX=RMAX, dr=dr)
    return stacked_profile/nvoids
