import numpy as np
import treecorr
from time import sleep
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.319, Ob0 = 0.049)
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

def make_randoms(ra, dec, z, size_random, col_names=['ra','dec','g1','g2','z']):

    ra_rand = np.random.uniform(min(ra), max(ra), size_random)
    sindec_rand = np.random.uniform(np.sin(min(dec*np.pi/180)), np.sin(max(dec*np.pi/180)), size_random)
    dec_rand = np.arcsin(sindec_rand)*(180/np.pi)

    y,xbins  = np.histogram(z, 25)
    x  = xbins[:-1]+0.5*np.diff(xbins)
    n = 3
    poly = np.polyfit(x,y,n)
    zr = np.random.uniform(z.min(),z.max(),1000000)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = np.random.choice(zr,len(ra_rand),replace=True,p=peso)

    #z_rand = np.random.choice(z,size=len(ra_rand),replace=True)

    d = {col_names[0]: ra_rand, col_names[1]: dec_rand, col_names[-1]:z_rand}
    randoms = d
    #randoms = pd.DataFrame(data = d)

    return randoms


def make_randoms_box(x, y, z, size_random, col_names=['x','y','z']):
    '''
    N = int(size_random**(1./3.))+1
    x_rand = np.random.uniform(min(x), max(x), N)
    y_rand = np.random.uniform(min(y), max(y), N)
    z_rand = np.random.uniform(min(z), max(z), N)
    '''
    xv, yv, zv = np.random.randint(x.min(), x.max(), size=(3,size_random))
    
    randoms = {col_names[0]: xv, col_names[1]: yv, col_names[2]:zv}

    return randoms


def compute_xi_2d(positions, random_positions,
                  npi = 16, nbins = 12,
                  rmin = 0.1, rmax = 100., pi_max = 60.,
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','z'],
                  cosmo = cosmo, ncores = 4,box=False):

    

    """ Compute the galaxy-shape correlation in 3D. """

    # arrays to store the output
    r         = np.zeros(nbins)
    mean_r    = np.zeros(nbins)
    mean_logr = np.zeros(nbins)

    xi    = np.zeros((npi, nbins))
    xi_jk = np.zeros((NPatches, npi, nbins))
    dd_jk = np.zeros_like(xi_jk)
    dr_jk = np.zeros_like(xi_jk)
    rr_jk = np.zeros_like(xi_jk)


    # catalogues
    if box:
        x, y, z = col_names
        pcat = treecorr.Catalog(x=positions[x], 
                                y=positions[y],
                                z=positions[z], 
                                npatch = NPatches
                               )
    
        rcat = treecorr.Catalog(x=random_positions[x], 
                                y=random_positions[y],
                                z=random_positions[z],
                                npatch = NPatches,
                                patch_centers = pcat.patch_centers
                               )

    else:
        ra, dec, z = col_names
        d_p  = cosmo.comoving_distance(positions[z]).value
        d_r = cosmo.comoving_distance(random_positions[z]).value
        
        pcat = treecorr.Catalog(ra=positions[ra], dec=positions[dec],
                                 r=d_p, npatch = NPatches,
                                 ra_units='deg', dec_units='deg')
    
        rcat = treecorr.Catalog(ra=random_positions[ra], dec=random_positions[dec],
                                 r=d_r, npatch = NPatches,
                                 patch_centers = pcat.patch_centers,
                                 ra_units='deg', dec_units='deg')
        


    Nd = pcat.sumw
    Nr = rcat.sumw
    NNpairs = (Nd*(Nd - 1))/2.
    RRpairs = (Nr*(Nr - 1))/2.
    NRpairs = (rcat.sumw*pcat.sumw)

    f0 = RRpairs/NNpairs
    f1 = RRpairs/NRpairs

    Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
    pibins = zip(Pi[:-1],Pi[1:])

    # now loop over Pi bins, and compute w(r_p | Pi)
    #bar = progressbar.ProgressBar(maxval=npi-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    for p,(plow,phigh) in enumerate(pibins):

        #bar.update(p)
        sleep(0.1)

        dd = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
                                    min_rpar=plow, max_rpar=phigh,
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        dr = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
                                    min_rpar=plow, max_rpar=phigh,
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        rr = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
                                    min_rpar=plow, max_rpar=phigh,
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')

        dd.process(pcat,pcat, metric='Rperp', num_threads = ncores)
        dr.process(pcat,rcat, metric='Rperp', num_threads = ncores)
        rr.process(rcat,rcat, metric='Rperp', num_threads = ncores)

        r[:] = np.copy(dd.rnom)
        mean_r[:] = np.copy(dd.meanr)
        mean_logr[:] = np.copy(dd.meanlogr)

        xi[p, :] = (dd.weight*0.5*f0 - (2.*dr.weight)*f1 + rr.weight*0.5) / (rr.weight*0.5)

        #xi[p,:], var = dd.calculateXi(rr=rr,dr=dr)

        #Here I compute the variance
        func = lambda corrs: corrs[0].weight*0.5
        func2 = lambda corrs: corrs[0].weight
        dd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dd], 'jackknife', func = func)
        dr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dr], 'jackknife', func = func2)
        rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func)

        dd.finalize()
        dr.finalize()
        rr.finalize()

    for i in range(NPatches):

        swd = np.sum(pcat.w[~(pcat.patch == i)])
        swr = np.sum(rcat.w[~(rcat.patch == i)])

        NNpairs_JK = (swd*(swd - 1))/2.
        RRpairs_JK = (swr*(swr - 1))/2.
        NRpairs_JK = (swd*swr)

        xi_jk[i, :, :] = (dd_jk[i, :, :]/NNpairs_JK - (2.*dr_jk[i, :, :])/NRpairs_JK + rr_jk[i, :, :]/RRpairs_JK) / (rr_jk[i, :, :]/RRpairs_JK)

    xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
    xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null

    xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5

    return r, mean_r, mean_logr, xPi, xi, xi_jk

def compute_wprp(positions, random_positions,
                  npi = 16, nbins = 12,
                  rmin = 0.1, rmax = 100., pi_max = 60.,
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','z'],
                  cosmo = cosmo, ncores = 4,box=False):
    
    r, mean_r, mean_logr, xPi, xi, xi_jk = compute_xi_2d(positions,random_positions,
                                                         npi, nbins, rmin, rmax, pi_max,
                                                         NPatches, slop, col_names,
                                                         cosmo, ncores,box)

    wprp = np.trapz(xi,xPi,axis=0)
    wprp_JK = np.trapz(xi_jk,xPi,axis=1)
    wprp_mean = np.mean(wprp_JK, axis = 0)
    wprp_diff = wprp_JK - wprp_mean
    cov_JK = ((NPatches-1)/NPatches)*np.sum(np.einsum('ij,ik->ijk',wprp_diff,wprp_diff), axis = 0)
    sigma = np.sqrt(np.diagonal(cov_JK))

    return r, wprp, sigma, cov_JK, xPi, xi, xi_jk, wprp_JK


def compute_input_randoms(random_positions,
                  npi = 16, nbins = 12, 
                  rmin = 0.1, rmax = 100., pi_max = 60.,
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','z'],
                  cosmo = cosmo, ncores = 4):
    
    ra, dec, z = col_names
    
    """ Compute the galaxy-shape correlation in 3D. """
    
    # arrays to store the output
    rr_corr =[]
    rr_jk = np.zeros((NPatches, npi, nbins))
        
    d_r = cosmo.comoving_distance(random_positions[z]).value
    
    # catalogues
    rcat = treecorr.Catalog(ra=random_positions[ra], dec=random_positions[dec], 
                             r=d_r, npatch = NPatches, 
                             ra_units='deg', dec_units='deg')        

    Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
    pibins = zip(Pi[:-1],Pi[1:])

    # now loop over Pi bins, and compute w(r_p | Pi)
    #bar = progressbar.ProgressBar(maxval=npi-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    for p,(plow,phigh) in enumerate(pibins):
        
        #bar.update(p)
        sleep(0.1)

        rr = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax, 
                                    min_rpar=plow, max_rpar=phigh, 
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        rr.process(rcat,rcat, metric='Rperp', num_threads = ncores)

        rr_corr += [rr]
        
        #Here I compute the variance
        func = lambda corrs: corrs[0].weight*0.5
        rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func)
        
        rr.finalize()
    
    return rcat, rr_corr, rr_jk

def compute_xi_2d_with_input_randoms(positions, randoms,
                  npi = 16, nbins = 12,
                  rmin = 0.1, rmax = 100., pi_max = 60.,
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','z'],
                  cosmo = cosmo, ncores = 4):

    rcat, rr_corr, rr_jk = randoms
    
    ra, dec, z = col_names

    """ Compute the galaxy-shape correlation in 3D. """

    # arrays to store the output
    r         = np.zeros(nbins)
    mean_r    = np.zeros(nbins)
    mean_logr = np.zeros(nbins)

    xi    = np.zeros((npi, nbins))
    xi_jk = np.zeros((NPatches, npi, nbins))
    dd_jk = np.zeros_like(xi_jk)
    dr_jk = np.zeros_like(xi_jk)
    

    d_p  = cosmo.comoving_distance(positions[z]).value

    # catalogues
    pcat = treecorr.Catalog(ra=positions[ra], dec=positions[dec],
                             r=d_p, npatch = NPatches,
                             patch_centers = rcat.patch_centers,
                             ra_units='deg', dec_units='deg')


    Nd = pcat.sumw
    Nr = rcat.sumw
    NNpairs = (Nd*(Nd - 1))/2.
    RRpairs = (Nr*(Nr - 1))/2.

    f = RRpairs/NNpairs


    Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
    pibins = zip(Pi[:-1],Pi[1:])

    # now loop over Pi bins, and compute w(r_p | Pi)
    #bar = progressbar.ProgressBar(maxval=npi-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    for p,(plow,phigh) in enumerate(pibins):

        #bar.update(p)
        sleep(0.1)

        dd = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
                                    min_rpar=plow, max_rpar=phigh,
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')

        dd.process(pcat,pcat, metric='Rperp', num_threads = ncores)

        r[:] = np.copy(dd.rnom)
        mean_r[:] = np.copy(dd.meanr)
        mean_logr[:] = np.copy(dd.meanlogr)
        
        xi[p, :] = (dd.weight/NNpairs)/((rr_corr[p]).weight/RRpairs) - 1.


        #xi[p,:], var = dd.calculateXi(rr=rr,dr=dr)

    #Here I compute the variance
        func = lambda corrs: corrs[0].weight*0.5

        dd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dd], 'jackknife', func = func)


        dd.finalize()

    for i in range(NPatches):

        swd = np.sum(pcat.w[~(pcat.patch == i)])
        swr = np.sum(rcat.w[~(rcat.patch == i)])

        NNpairs_JK = (swd*(swd - 1))/2.
        RRpairs_JK = (swr*(swr - 1))/2.
        
        xi_jk[i, :, :] = (dd_jk[i, :, :]/NNpairs_JK) / (rr_jk[i, :, :]/RRpairs_JK) - 1.
        
    xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
    xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null

    xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5

    return r, mean_r, mean_logr, xPi, xi, xi_jk


def compute_wprp_with_input_randoms(positions, randoms,
                  npi = 16, nbins = 12, 
                  rmin = 0.1, rmax = 100., pi_max = 60.,
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','z'],
                  cosmo = cosmo, ncores = 4):
    
    r, mean_r, mean_logr, xPi, xi, xi_jk = compute_xi_2d_with_input_randoms(positions,randoms,
                                                         npi, nbins, rmin, rmax, pi_max,
                                                         NPatches, slop, col_names,
                                                         cosmo, ncores)
    
    wprp = np.trapz(xi,xPi,axis=0)
    wprp_JK = np.trapz(xi_jk,xPi,axis=1)
    wprp_mean = np.mean(wprp_JK, axis = 0)
    wprp_diff = wprp_JK - wprp_mean
    cov_JK = ((NPatches-1)/NPatches)*np.sum(np.einsum('ij,ik->ijk',wprp_diff,wprp_diff), axis = 0)
    sigma = np.sqrt(np.diagonal(cov_JK))   
    
    return r, wprp, sigma, cov_JK, xPi, xi, xi_jk, wprp_JK


def compute_gp_2d(positions, shapes, 
                  random_positions, random_shapes, 
                  npi = 16, nbins = 12, 
                  rmin = 0.1, rmax = 100., pi_max = 60.,                  
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','g1','g2','z'],
                  cosmo = cosmo, ncores = 4):
    
    """ Compute the galaxy-shape correlation in 3D. """
    ra, dec, g1, g2, z = col_names
    
    # arrays to store the output
    r     = np.zeros(nbins)
    mean_r     = np.zeros(nbins)
    mean_logr     = np.zeros(nbins)
    xi = np.zeros((npi, nbins))
    xi_jk = np.zeros((NPatches, npi, nbins))
    xi_ng_jk = np.zeros_like(xi_jk)
    xi_rg_jk = np.zeros_like(xi_jk)
    rr_weight_jk = np.zeros_like(xi_jk)
    f0_JK = np.zeros(NPatches)
    f1_JK = np.zeros(NPatches)
    
    # catalogues
    d_p  = cosmo.comoving_distance(positions[z]).value
    d_s  = cosmo.comoving_distance(shapes[z]).value
    d_pr = cosmo.comoving_distance(random_positions[z]).value
    d_sr = cosmo.comoving_distance(random_shapes[z]).value
    
    # catalogues
    pcat  = treecorr.Catalog(ra=positions[ra], dec=positions[dec], 
                             r=d_p, npatch = NPatches, 
                             ra_units='deg', dec_units='deg')
    
    scat  = treecorr.Catalog(g1=shapes[g1], g2=shapes[g2], 
                             ra=shapes[ra], dec=shapes[dec], 
                             r=d_s, npatch = NPatches, 
                             patch_centers = pcat.patch_centers, 
                             ra_units='deg', dec_units='deg')
    
    rpcat = treecorr.Catalog(ra=random_positions[ra], dec=random_positions[dec], 
                             r=d_pr, npatch = NPatches, 
                             patch_centers = pcat.patch_centers, 
                             ra_units='deg', dec_units='deg')
    
    rscat = treecorr.Catalog(ra=random_shapes[ra], dec=random_shapes[dec], 
                             r=d_sr, npatch = NPatches, 
                             patch_centers = pcat.patch_centers, 
                             ra_units='deg', dec_units='deg')
    

    # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
    NGtot = pcat.sumw*scat.sumw 
    RRtot = rpcat.sumw*rscat.sumw 
    RGtot = rpcat.sumw*scat.sumw
    f0 = RRtot / NGtot
    f1 = RRtot / RGtot

    Pi = np.linspace(-100, 100, npi+1)
    pibins = zip(Pi[:-1],Pi[1:])

    # now loop over Pi bins, and compute w(r_p | Pi)
    for p,(plow,phigh) in enumerate(pibins):
        
        rr = treecorr.NNCorrelation(nbins=nbins, min_sep=0.1, max_sep=10, min_rpar=plow, max_rpar=phigh, bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        ng = treecorr.NGCorrelation(nbins=nbins, min_sep=0.1, max_sep=10, min_rpar=plow, max_rpar=phigh, bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        rg = treecorr.NGCorrelation(nbins=nbins, min_sep=0.1, max_sep=10, min_rpar=plow, max_rpar=phigh, bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')  
                
        ng.process(pcat, scat, metric='Rperp', num_threads = ncores)
        ng_xi = ng.xi
        ng_weight = ng.weight

        rg.process(rpcat, scat, metric='Rperp', num_threads = ncores)
        rg_xi = rg.xi  
        rg_weight = rg.weight

        rr.process(rpcat, rscat, metric='Rperp', num_threads = ncores)
        rr_weight = rr.weight
        r[:] = np.copy(rr.rnom)
        mean_r[:] = np.copy(rr.meanr)
        mean_logr[:] = np.copy(rr.meanlogr)

        xi[p, :] = (f0 * (ng.xi * ng.weight) - f1 * (rg.xi * rg.weight) ) / rr.weight
    
        #Here I compute the variance
        func_ng = lambda corrs: corrs[0].xi * corrs[0].weight
        func_rg = lambda corrs: corrs[0].xi * corrs[0].weight
        func_rr = lambda corrs: corrs[0].weight
        xi_ng_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([ng], 'jackknife', func = func_ng)
        xi_rg_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rg], 'jackknife', func = func_rg)
        rr_weight_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)

    for i in range(NPatches):
        cond1 = pcat.patch == i
        cond2 = scat.patch == i
        cond3 = rpcat.patch == i
        cond4 = rscat.patch == i
        swd1 = np.sum(pcat.w[~cond1])
        swd2 = np.sum(scat.w[~cond2])
        swr1 = np.sum(rpcat.w[~cond3])
        swr2 = np.sum(rscat.w[~cond4])
        pcat_ra = pcat.ra[~cond1]
        scat_ra = scat.ra[~cond2]
        rpcat_ra = rpcat.ra[~cond3]
        rscat_ra = rscat.ra[~cond4]
        
        NGtot_JK = swd1*swd2 
        RRtot_JK = swr1*swr2 
        RGtot_JK = swr1*swd2           
        f0_JK[i] = RRtot_JK / NGtot_JK
        f1_JK[i] = RRtot_JK / RGtot_JK
        xi_jk[i, :, :] = (f0_JK[i] * xi_ng_jk[i, :, :] - f1_JK[i] * xi_rg_jk[i, :, :]) / rr_weight_jk[i, :, :]

    xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
    xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null

    xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
    
    return r, mean_r, mean_logr, xPi, xi, xi_jk

def compute_wgp(positions, shapes, 
                randoms_positions, randoms_shapes, 
                npi = 16, nbins = 12,
                rmin = 0.1, rmax = 100., pi_max = 60.,
                NPatches = 16, slop = 0.,
                col_names=['ra','dec','g1','g2','z'],
                cosmo = cosmo, ncores = 4):
    
    r, mean_r, mean_logr, xPi, xi, xi_jk = compute_gp_2d(positions, shapes, randoms_positions, randoms_shapes, npi, nbins, rmin, rmax, pi_max, NPatches, slop, col_names,cosmo, ncores)
    wgp = np.trapz(xi,xPi,axis=0)
    wgp_JK = np.trapz(xi_jk,xPi,axis=1)
    wgp_mean = np.mean(wgp_JK, axis = 0)
    wgp_diff = wgp_JK - wgp_mean
    cov_JK = ((NPatches-1)/NPatches)*np.sum(np.einsum('ij,ik->ijk',wgp_diff,wgp_diff), axis = 0)
    sigma = np.sqrt(np.diagonal(cov_JK))   
    
    return r, wgp, sigma, cov_JK, xi_jk, wgp_JK


def compute_gp_r_mu(positions, shapes, 
                  random_positions, random_shapes, 
                  npi = 16, nbins = 12, 
                  rmin = 0.1, rmax = 100., pi_max = 60.,                  
                  NPatches = 16, slop = 0.,
                  col_names=['ra','dec','g1','g2','z'],
                  cosmo = cosmo, ncores = 4):
    
    """ Compute the galaxy-shape correlation in 3D. """
    ra, dec, g1, g2, z = col_names
    
    # arrays to store the output
    r     = np.zeros(nbins)
    mean_r     = np.zeros(nbins)
    mean_logr     = np.zeros(nbins)
    xi = np.zeros((npi, nbins))
    xi_jk = np.zeros((NPatches, npi, nbins))
    xi_ng_jk = np.zeros_like(xi_jk)
    xi_rg_jk = np.zeros_like(xi_jk)
    rr_weight_jk = np.zeros_like(xi_jk)
    f0_JK = np.zeros(NPatches)
    f1_JK = np.zeros(NPatches)
    
    # catalogues
    d_p  = cosmo.comoving_distance(positions[z]).value
    d_s  = cosmo.comoving_distance(shapes[z]).value
    d_pr = cosmo.comoving_distance(random_positions[z]).value
    d_sr = cosmo.comoving_distance(random_shapes[z]).value
    
    # catalogues
    pcat  = treecorr.Catalog(ra=positions[ra], dec=positions[dec], 
                             r=d_p, npatch = NPatches, 
                             ra_units='deg', dec_units='deg')
    
    scat  = treecorr.Catalog(g1=shapes[g1], g2=shapes[g2], 
                             ra=shapes[ra], dec=shapes[dec], 
                             r=d_s, npatch = NPatches, 
                             patch_centers = pcat.patch_centers, 
                             ra_units='deg', dec_units='deg')
    
    rpcat = treecorr.Catalog(ra=random_positions[ra], dec=random_positions[dec], 
                             r=d_pr, npatch = NPatches, 
                             patch_centers = pcat.patch_centers, 
                             ra_units='deg', dec_units='deg')
    
    rscat = treecorr.Catalog(ra=random_shapes[ra], dec=random_shapes[dec], 
                             r=d_sr, npatch = NPatches, 
                             patch_centers = pcat.patch_centers, 
                             ra_units='deg', dec_units='deg')
    

    # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
    NGtot = pcat.sumw*scat.sumw 
    RRtot = rpcat.sumw*rscat.sumw 
    RGtot = rpcat.sumw*scat.sumw
    f0 = RRtot / NGtot
    f1 = RRtot / RGtot

    Pi = np.linspace(-100, 100, npi+1)
    pibins = zip(Pi[:-1],Pi[1:])

    # now loop over Pi bins, and compute w(r_p | Pi)
    for p,(plow,phigh) in enumerate(pibins):
        
        rr = treecorr.NNCorrelation(nbins=nbins, min_sep=0.1, max_sep=10, min_rpar=plow, max_rpar=phigh, bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        ng = treecorr.NGCorrelation(nbins=nbins, min_sep=0.1, max_sep=10, min_rpar=plow, max_rpar=phigh, bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        rg = treecorr.NGCorrelation(nbins=nbins, min_sep=0.1, max_sep=10, min_rpar=plow, max_rpar=phigh, bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')  
                
        ng.process(pcat, scat, metric='Euclidean', num_threads = ncores)
        ng_xi = ng.xi
        ng_weight = ng.weight

        rg.process(rpcat, scat, metric='Euclidean', num_threads = ncores)
        rg_xi = rg.xi  
        rg_weight = rg.weight

        rr.process(rpcat, rscat, metric='Euclidean', num_threads = ncores)
        rr_weight = rr.weight
        r[:] = np.copy(rr.rnom)
        mean_r[:] = np.copy(rr.meanr)
        mean_logr[:] = np.copy(rr.meanlogr)

        xi[p, :] = (f0 * (ng.xi * ng.weight) - f1 * (rg.xi * rg.weight) ) / rr.weight
    
        #Here I compute the variance
        func_ng = lambda corrs: corrs[0].xi * corrs[0].weight
        func_rg = lambda corrs: corrs[0].xi * corrs[0].weight
        func_rr = lambda corrs: corrs[0].weight
        xi_ng_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([ng], 'jackknife', func = func_ng)
        xi_rg_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rg], 'jackknife', func = func_rg)
        rr_weight_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)

    for i in range(NPatches):
        cond1 = pcat.patch == i
        cond2 = scat.patch == i
        cond3 = rpcat.patch == i
        cond4 = rscat.patch == i
        swd1 = np.sum(pcat.w[~cond1])
        swd2 = np.sum(scat.w[~cond2])
        swr1 = np.sum(rpcat.w[~cond3])
        swr2 = np.sum(rscat.w[~cond4])
        pcat_ra = pcat.ra[~cond1]
        scat_ra = scat.ra[~cond2]
        rpcat_ra = rpcat.ra[~cond3]
        rscat_ra = rscat.ra[~cond4]
        
        NGtot_JK = swd1*swd2 
        RRtot_JK = swr1*swr2 
        RGtot_JK = swr1*swd2           
        f0_JK[i] = RRtot_JK / NGtot_JK
        f1_JK[i] = RRtot_JK / RGtot_JK
        xi_jk[i, :, :] = (f0_JK[i] * xi_ng_jk[i, :, :] - f1_JK[i] * xi_rg_jk[i, :, :]) / rr_weight_jk[i, :, :]

    xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
    xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null

    xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
    
    return r, mean_r, mean_logr, xPi, xi, xi_jk
