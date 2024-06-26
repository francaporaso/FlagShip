import numpy as np
from astropy.io import fits
import time

from tools.void import Void, StackedVoid, Tracer
from tools.save_file import save_file

# leer cat de Voids -> convertirlos en Void() -> leer cat de trazadores -> calcular perfiles -> apilarlos

def lens_cat(folder, lcat, 
             Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG):
    
    L = np.loadtxt(folder+lcat).T

    Rv    = L[1]
    z     = L[4]
    rho_1 = L[8] #Sobredensidad integrada a un radio de void 
    rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
    flag  = L[11]

    mask = ((Rv >= Rv_min)&(Rv < Rv_max))&((z >= z_min)&(z < z_max))&(
        (rho_1 >= rho1_min)&(rho_1 < rho1_max))&((rho_2 >= rho2_min)&(rho_2 < rho2_max))&(flag >= FLAG)        

    L = L[:,mask]

    return L

def method1(tcat,
            xv,yv,zv,rv,
            RMIN,RMAX,dr):
    '''making void objects all at once and storing in voids_list'''

    voids_list = list(map(Void, xv,yv,zv,rv))
    delta_list = np.array([v.radial_density_profile(cat=tcat, RMIN=RMIN, RMAX=RMAX, dr=dr) for v in voids_list])

    stacked_profile = np.mean(delta_list, axis=0)
    return stacked_profile

def method2(tcat,
            xv,yv,zv,rv,
            RMIN,RMAX,dr):
    '''making void objects one by one and overwriting v'''

    NBINS = int(round(((RMAX-RMIN)/dr),0))
    nvoids = len(xv)

    stacked_profile = np.zeros(NBINS)
    for i in range(nvoids):
        v = Void(xv[i], yv[i], zv[i], rv[i])
        stacked_profile += v.radial_density_profile(cat=tcat, RMIN=RMIN, RMAX=RMAX, dr=dr)
    stacked_profile /= nvoids

    return stacked_profile
    
def method3(tcat,
            xv,yv,zv,rv,
            RMIN,RMAX,dr):
    '''creating a void object with all tracers from the individual voids'''
    nvoids = len(xv)

    tr_list = []

    for i in range(nvoids):
        v = Void(xv[i], yv[i], zv[i], rv[i])
        v.get_tracers(cat=tcat, RMAX=RMAX, center=True)
        tr_list.append(v.tr)
    ## as tr_list is a list of lists, we flatten
    tr_flat = [x for xs in tr_list for x in xs] 

    stacked_void = StackedVoid(tr_flat)
    stacked_profile = stacked_void.radial_density_profile(RMIN=RMIN, RMAX=RMAX, dr=dr)

    return stacked_profile

def main(tfolder, tcat,
         RMIN, RMAX, dr,
         lfolder, lcat, 
         Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG,
         sample):

    print('Loading catalogs...')
    tcat = fits.open(tfolder+tcat)[1].data # catalog of tracers
    L = lens_cat(lfolder, lcat, 
                 Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG)

    xv, yv, zv, rv = L[5], L[6], L[7], L[1]
    print(f'Nvoids: {len(L[1])}')

    # print('Running method1...')
    # t_in = time.time()
    # stacked_profile_1 = method1(tcat=tcat,xv=xv,yv=yv,zv=zv,rv=rv,RMIN=RMIN,RMAX=RMAX,dr=dr)
    stacked_profile_1 = 0
    # print(f'Ended in {time.time()-t_in} s')

    # t_in = time.time()
    # print('Running method2...')
    # stacked_profile_2 = method2(tcat=tcat,xv=xv,yv=yv,zv=zv,rv=rv,RMIN=RMIN,RMAX=RMAX,dr=dr)
    stacked_profile_2 = 0
    # print(f'Ended in {time.time()-t_in} s')
    
    t_in = time.time()
    print('Running method3...')
    stacked_profile_3 = method3(tcat=tcat,xv=xv,yv=yv,zv=zv,rv=rv,RMIN=RMIN,RMAX=RMAX,dr=dr)
    print(f'Ended in {time.time()-t_in} s')

    #save file
    folder = f'profiles/radial/'
    filename = f'sp-{sample}_r{int(Rv_min)}-{int(Rv_max)}-z0{int(z_min*10)}_{int(z_max*10)}.fits'
    print(f'Saving file in {folder+filename}')
    save_file([stacked_profile_1, stacked_profile_2, stacked_profile_3], folder, filename)

    return True

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='pru')
    parser.add_argument('-tcat', action='store', dest='tcat',default='pru')
    parser.add_argument('-lcat', action='store', dest='lcat',default='pru')
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

    args = parser.parse_args()
    sample = args.sample
    tcat = args.tcat
    lcat = args.lcat
    RMIN = float(args.RMIN)
    RMAX = float(args.RMAX)
    dr   = float(args.dr) #space between shells, in units of void radius
    Rv_min   = float(args.Rv_min)
    Rv_max   = float(args.Rv_max) 
    z_min    = float(args.z_min) 
    z_max    = float(args.z_max) 
    rho1_min = float(args.rho1_min)
    rho1_max = float(args.rho1_max) 
    rho2_min = float(args.rho2_min)
    rho2_max = float(args.rho2_max) 

    tfolder = '' #folder of the tracers cat
    lfolder = '' #folder of the lenses cat

    FLAG = 2

    tini = time.time()
    S = main(tfolder, tcat,
             RMIN, RMAX, dr,
             lfolder, lcat, 
             Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG,
             sample)
    tfin = time.time()
    if S:
        print(f'Job finished succesfully in {tfin-tini} s!')