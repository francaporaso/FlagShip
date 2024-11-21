import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM

def make_randoms(ra, dec, z, size_random):
    
    print('Making randoms...')

    dec = np.deg2rad(dec)
    sindec_rand = np.random.uniform(np.sin(dec.min()), np.sin(dec.max()), size_random)
    dec_rand = np.arcsin(sindec_rand)*(180/np.pi)
    ra_rand  = np.random.uniform(ra.min(), ra.max(), size_random)

    y,xbins  = np.histogram(z, 25)
    x  = xbins[:-1]+0.5*np.diff(xbins)
    n = 3
    poly = np.polyfit(x,y,n)
    zr = np.random.uniform(z.min(),z.max(),1_000_000)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = np.random.choice(zr,size_random,replace=True,p=peso)

    randoms = {'ra': ra_rand, 'dec': dec_rand, 'z':z_rand}
    
    print('Wii randoms!')
    
    return randoms

def ang2xyz(ang_pos):
    ang_pos['ra']
    ang_pos['dec']
    ang_pos['z']