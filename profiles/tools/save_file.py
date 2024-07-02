# guardar archivos
from astropy.io import fits
import numpy as np

### pasar 2 dicts, for loop para header.append((key1, value1)) y for loop para generar table con dict2 

def save_file(folder, filename,
              stacked_profile, nvoids, RMIN, RMAX, dr,
              Rv_min, Rv_max, z_min, z_max):

    header = fits.Header()
    header.append(('nvoids', nvoids))
    header.append(('rv_min', Rv_min))
    header.append(('rv_max', Rv_max))
    header.append(('z_min', z_min))
    header.append(('z_max', z_max))

    NBINS = int(round((RMAX-RMIN)/dr,0))
    bines = np.linspace(RMIN,RMAX,num=NBINS+1)
    R = (bines[:-1] + np.diff(bines)*0.5)

    table = [fits.Column(name='R', format='E', array=R),
             fits.Column(name='stacked_profile', format='E', array=stacked_profile),
            ]

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu, tbhdu])
    hdul.writeto(folder+filename, overwrite=True)

    return True