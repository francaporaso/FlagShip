import numpy as np
from astropy.io import fits


def uniform_cat(lencat=100_000, l=1000., lm_min=9., lm_max=15.):

    xhalo = np.random.uniform(-l,l,lencat)
    yhalo = np.random.uniform(-l,l,lencat)
    zhalo = np.random.uniform(-l,l,lencat)
    lmhalo = np.random.uniform(lm_min,lm_max,lencat)
    flag_central = np.zeros(lencat)

    header = fits.Header()
    header.append( ('lencat', lencat) )
    header.append( ('L', l) )
    header.append( ('lm_min', lm_min) )
    header.append( ('lm_max', lm_max) )

    tabla = [
        fits.Column(name='xhalo', format='E', array=xhalo),
        fits.Column(name='yhalo', format='E', array=yhalo),
        fits.Column(name='zhalo', format='E', array=zhalo),
        fits.Column(name='lmhalo', format='E', array=lmhalo),
        fits.Column(name='flag_central', format='E', array=flag_central),
    ]

    t = fits.BinTableHDU.from_columns(fits.ColDefs(tabla))
    hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([hdu, t])

    hdul.writeto('tests/testcat_uniform.fits', overwrite=True)

def normal_cat(lencat=100_000, sigma=0., mu=300, lm_min=9., lm_max=15.):

    xhalo = np.random.normal(sigma,mu,lencat)
    yhalo = np.random.normal(sigma,mu,lencat)
    zhalo = np.random.normal(sigma,mu,lencat)
    lmhalo = np.random.uniform(lm_min,lm_max,lencat)

    l = np.array([xhalo, yhalo, zhalo]).flatten().max()

    header = fits.Header()
    header.append( ('lencat', lencat) )
    header.append( ('L', l) )
    header.append( ('lm_min', lm_min) )
    header.append( ('lm_max', lm_max) )

    tabla = [
        fits.Column(name='xhalo', format='E', array=xhalo),
        fits.Column(name='yhalo', format='E', array=yhalo),
        fits.Column(name='zhalo', format='E', array=zhalo),
        fits.Column(name='lmhalo', format='E', array=lmhalo),
            ]

    t = fits.BinTableHDU.from_columns(fits.ColDefs(tabla))
    hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([hdu, t])

    hdul.writeto('tests/testcat_normal.fits', overwrite=True)


def lens_cat(nvoids=100, l=800., rv_min=6., rv_max=30.):

    import pandas as pd

    rmean = (rv_max+rv_min)/2
    rdisp = (rv_max-rv_min)/2

    xvoid = np.random.uniform(-l,l,nvoids)
    yvoid = np.random.uniform(-l,l,nvoids)
    zvoid = np.random.uniform(-l,l,nvoids)
    rvoid = np.random.normal(rmean,rdisp,nvoids)

    redshift = np.random.uniform(low=0.1,high=0.5,size=nvoids)
    rho1 = np.random.uniform(low=-1.,high=1.,size=nvoids)
    rho2 = np.random.uniform(low=-1.,high=100.,size=nvoids)
    flag = np.full(nvoids, 2)
    
    ## .txt
    df = pd.DataFrame({
        '#0':0, #0
        'rvoid': rvoid, #1
        '#2':0, #2
        '#3':0, #3
        'redshift': redshift, #4
        'xvoid': xvoid, #5
        'yvoid': yvoid, #6
        'zvoid': zvoid, #7
        'rho1' : rho1, #8
        'rho2' : rho2, #9
        '#10':0, #10
        'flag':flag, #11
    })

    df.to_csv('tests/testcat_lens.txt', sep='\t', index=False)

    ## .fits
    # header = fits.Header()
    # header.append( ('nvoids', nvoids) )
    # header.append( ('rv_min', rv_min) )
    # header.append( ('rv_max', rv_max) )

    # tabla = [
    #     fits.Column(name='xvoid', format='E', array=xvoid),
    #     fits.Column(name='yvoid', format='E', array=yvoid),
    #     fits.Column(name='zvoid', format='E', array=zvoid),
    #     fits.Column(name='rvoid', format='E', array=rvoid),
    #         ]

    # t = fits.BinTableHDU.from_columns(fits.ColDefs(tabla))
    # hdu = fits.PrimaryHDU(header=header)
    # hdul = fits.HDUList([hdu, t])

    # hdul.writeto('tests/testcat_lens.fits', overwrite=True)



if __name__ == '__main__':
    uniform_cat()