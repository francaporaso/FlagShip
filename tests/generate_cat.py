import numpy as np
from astropy.io import fits

#x,y,z,m
lencat = 100_000
l = 1000. #box size
lm_min = 9.
lm_max = 15.

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

hdul.writeto('tests/testcat.fits', overwrite=True)