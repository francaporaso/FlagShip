import numpy as np
from astropy.io import fits


def uniform_cat(lencat=100_000, l=1000., lm_min=9., lm_max=15.):

    xhalo = np.random.uniform(-l,l,lencat)
    yhalo = np.random.uniform(-l,l,lencat)
    zhalo = np.random.uniform(-l,l,lencat)
    lmhalo = np.random.uniform(lm_min,lm_max,lencat)

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


if __name__ == '__main__':
    normal_cat()