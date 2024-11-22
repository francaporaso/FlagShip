import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM

h = 1.0
cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)

## TODO puede que sea más efficiente simplemente pasando los maximos y minimos
## para z no funcionaría xq tiene q interpolar...
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

def ang2xyz(ra, dec, redshift,
            cosmo=cosmo):

    comdist = cosmo.comoving_distance(redshift).value #Mpc; Mpc/h si h=1
    x = comdist * np.cos(dec) * np.cos(ra)
    y = comdist * np.cos(dec) * np.sin(ra)
    z = comdist * np.sin(dec)

    return x,y,z

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    N = 10000
    ra  = 90.0 * np.random.rand(N)
    dec = np.rad2deg(np.sin(np.random.rand(N)))
    redshift = 0.4 * np.random.rand(N)
    rands_ang = make_randoms(ra,dec,redshift,10*N)
    rands_box = ang2xyz(*rands_ang.values())

    plt.hist(ra, bins=25, histtype='step')
    plt.hist(rands_ang['ra'], bins=25, histtype='step')
    plt.show()
    plt.hist(dec, bins=25, histtype='step')
    plt.hist(rands_ang['dec'], bins=25, histtype='step')
    plt.show()
    plt.hist(redshift, bins=25, histtype='step')
    plt.hist(rands_ang['z'], bins=25, histtype='step')
    plt.show()

    plt.scatter(ra*np.cos(np.deg2rad(dec)),dec,s=0.5)
    # plt.scatter(rands[0],rands[1],s=0.5)
    plt.show()