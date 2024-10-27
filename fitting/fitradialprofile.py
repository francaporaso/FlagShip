import numpy as np
import emcee
from functions import *


## TODO
## tiene mucho sentido hacer esto con Class; 
## algo así podría funcionar... pensar
## class Modelo:
##      logetc()
## class HSW(Modelo):
##      ...
def log_likelihood_HSW(theta, r, y, yerr):
    '''
    r : eje x
    y : datos eje y
    yerr: error en los datos -> L_S utiliza yerr como la inversa de la mat de cov
    '''
    rs,dc,a,b,x = theta
    modelo = HSW(r, rs, dc, a, b, x)
    
    L_S = -np.dot((y-modelo),np.dot(yerr,(y-modelo)))/2.0
        
    return L_S    

def log_prior_HSW(theta):
    rs,dc,a,b,x = theta
    if (0. <= rs <= 3.)&(-1. <= dc <= 0.)&(0. <= a <= 10.)&(1. <= b <= 20.)&(-10<=x<=10):
        return 0.0
    return -np.inf

def log_probability_HSW(theta, r, y, yerr):
    lp = log_prior_HSW(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_HSW(theta, r, y, yerr)

## TODO
## ver si se puede mejorar esto...
def ajuste(xdata, ydata, ycov, pos, log_probability,
           nit=1000, ncores=32, continue_run=False):
    
    '''
    ajuste con mcmc
    xdata: datos en el eje x
    ydata: datos en el eje y
    ycov: error de ydata, pueden ser errores de la diagonal o la matriz completa
    '''   

    nwalkers, ndim = pos.shape

    if ycov.shape == ydata.shape:
        yerr = ycov
        print('Usando diagonal')
    else:
        yerr = np.linalg.inv(ycov)
        print('Usando matriz de covarianza')

    backend = emcee.backends.HDFBackend('emcee_backends.h5')

    # with Pool(processes=ncores) as pool:
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata,ydata,yerr), pool=pool, backend=backend)
    #     sampler.run_mcmc(pos, nit, progress=True)


    ### PARA HAMAUS -> COMO LA FUNC ESTÁ PARALELIZADA NO PUEDE SER PARALELO EL AJUSTE
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xdata,ydata,yerr), backend=backend)

    if continue_run:
        print('continuando desde backend...')
        nit = nit - backend.iteration
        sampler.run_mcmc(None, nit, progress=True)
    else:
        backend.reset(nwalkers, ndim)
        sampler.run_mcmc(pos, nit, progress=True)

    mcmc_out = sampler.get_chain()

    return mcmc_out
