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
    x = comdist * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = comdist * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = comdist * np.sin(np.deg2rad(dec))

    return x,y,z

def plot_sky(ang_pos, rands_ang):
    
    plt.hist(ang_pos['ra'], bins=25, histtype='step', density=True)
    plt.hist(rands_ang['ra'], bins=25, histtype='step', density=True)
    plt.show()
    plt.hist(ang_pos['dec'], bins=25, histtype='step', density=True)
    plt.hist(rands_ang['dec'], bins=25, histtype='step', density=True)
    plt.show()
    plt.hist(ang_pos['z'], bins=25, histtype='step', density=True)
    plt.hist(rands_ang['z'], bins=25, histtype='step', density=True)
    plt.show()

    amin, amax = 0,90
    alpha = 0.9
    mran = (rands_ang['ra'] > amin) & (rands_ang['ra'] < amax) & (rands_ang['dec'] > amin) & (rands_ang['dec'] < amax)
    mtru = (ang_pos['ra'] > amin) & (ang_pos['ra'] < amax) & (ang_pos['dec'] > amin) & (ang_pos['dec'] < amax)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.scatter(rands_ang['ra'][mran] * np.cos(rands_ang['dec'][mran]*np.pi/180), rands_ang['dec'][mran], color='green', s=0.1, alpha=alpha)
    ax1.set_xlabel('RA * cos(Dec)')
    ax1.set_ylabel('Dec')
    ax1.set_title('Randoms')

    ax2.scatter(ang_pos['ra'][mtru] * np.cos(ang_pos['dec'][mtru]*np.pi/180), ang_pos['dec'][mtru], color='blue', s=0.1, alpha=alpha)
    ax2.set_xlabel('RA * cos(Dec)')
    ax2.set_ylabel('Dec')
    ax2.set_title('Data')

    plt.show()

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    N = 10_000
    z0,z1 = 0.1, 0.12

    with fits.open('/home/franco/FAMAF/Lensing/cats/MICE/mice18917.fits') as f:
        z_gal = f[1].data.z_cgal
        
        m_z = (z_gal < z1) & (z_gal >= z0)
        ra  = f[1].data.ra_gal[m_z]
        dec = f[1].data.dec_gal[m_z]
        z_gal = z_gal[m_z]
    
    if (s := m_z.sum()) < N:
        N = s
    print(N)

    ang_pos = {'ra':ra[:N], 'dec':dec[:N], 'z':z_gal[:N]}
    xyz_pos = ang2xyz(*ang_pos.values())
    rands_ang = make_randoms(*ang_pos.values(),10*N)
    rands_box = ang2xyz(*rands_ang.values())
    # mask = (np.abs(rands_box[2]) < 10)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*xyz_pos,
               s=1, alpha=0.3)
    plt.show()