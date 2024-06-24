import numpy as np
import astropy.io as fits

from tools.void import Void, Tracer
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

    nvoids = len(L.T)
    # print(f'Nvoids = {Nvoids}')

    return nvoids, L

def main(tfolder, tcat,
         RMIN, RMAX, dr,
         lfolder, lcat, 
         Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG,
         sample):

    tcat = fits.open(tfolder+tcat)[1].data # catalog of tracers

    nvoids, L = lens_cat(lfolder, lcat, 
                 Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG)
    
    xv, yv, zv, rv = L[5], L[6], L[7], L[1]

    voids_list = list(map(Void, xv,yv,zv,rv))
    delta_list = np.array([v.radial_density_profile(cat=tcat, RMIN=RMIN, RMAX=RMAX, dr=dr) for v in voids_list])

    stacked_profile = np.mean(delta_list, axis=0)

    #save file
    folder = f'profiles/radial/'
    filename = f'sp-{sample}_r{int(Rv_min)}-{int(Rv_max)}-z0{int(z_min*10)}_{int(z_max*10)}.fits'
    save_file(stacked_profile, folder, filename)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-sample', action='store', dest='sample',default='pru')
    parser.add_argument('-tcat', action='store', dest='sample',default='pru')
    parser.add_argument('-lcat', action='store', dest='sample',default='pru')
    parser.add_argument('-RMIN', action='store', dest='sample',default=0.01)
    parser.add_argument('-RMAX', action='store', dest='sample',default=5.0)
    parser.add_argument('-dr', action='store', dest='sample',default=0.25)
    parser.add_argument('-Rv_min', action='store', dest='sample',default=0.25)
    parser.add_argument('-Rv_max', action='store', dest='sample',default=0.25)
    parser.add_argument('-z_min', action='store', dest='sample',default=0.25)
    parser.add_argument('-z_max', action='store', dest='sample',default=0.25)
    parser.add_argument('-rho1_min', action='store', dest='sample',default=0.25)
    parser.add_argument('-rho1_max', action='store', dest='sample',default=0.25)
    parser.add_argument('-rho2_min', action='store', dest='sample',default=0.25)
    parser.add_argument('-rho2_max', action='store', dest='sample',default=0.25)

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

    main(tfolder, tcat,
         RMIN, RMAX, dr,
         lfolder, lcat, 
         Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, FLAG,
         sample)