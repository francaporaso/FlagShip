import numpy as np
from astropy.cosmology import LambdaCDM

def pm(z):
    """
    densidad media en Msun/(pc**2 Mpc)
    """
    h = 1.
    cosmo = LambdaCDM(H0 = 100.*h, Om0=0.3, Ode0=0.7)
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out

def HSW(r, rs, rv, delta, a, b):
    """
    function of Hamaus, Sutter and Wandelt 2015
    """
    d = delta*(1. - (r/rs)**a)/(1. + (r/rv)**b)
    return d

def LW(r,Rv,R2,dc,d2):
    """
    function of Lavaux and Wandelt 2012, modified
    """
    R_V = np.full_like(r, Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=R_V)*(dc + (d2-dc)*(r/Rv)**3) + ((r>R_V)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta

def HOH(r,Rv,R2,dc,d2):
    """
    function of Higuchi, Oguri and Takashi 2013
    also called double top-hat
    """

    unos = np.full_like(r,Rv)
    R2s  = np.full_like(r,R2)
    
    delta = (r<=unos)*dc + ((r>unos)&(r<=R2s))*d2 + (r>R2s)*0
    
    return delta

def chi_red(ajuste,data,err,gl):
	'''
	Reduced chi**2
	------------------------------------------------------------------
	INPUT:
	ajuste       (float or array of floats) fitted value/s
	data         (float or array of floats) data used for fitting
	err          (float or array of floats) error in data
	gl           (float) grade of freedom (number of fitted variables)
	------------------------------------------------------------------
	OUTPUT:
	chi          (float) Reduced chi**2 	
	'''
		
	BIN=len(data)
	chi=((((ajuste-data)**2)/(err**2)).sum())/float(BIN-1-gl)
	return chi