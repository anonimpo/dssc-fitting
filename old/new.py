import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import physical_constants


#parameters
#1 fix constant
e = physical_constants["elementary charge"][0]
k = physical_constants ["Boltzmann constant"][0]
T = 300                                 # K     ## room temperature 

# sample dependent
d =                                    ## thickness device
phi_0 =                                # flux photon ## sun 
alpha =                                # absorpsion sepctrum ? 

# to get : D_e, tau_e, n_0, m 


def j_total (D_e, tau_e, n_0, m, V):
    # from equation 19b
    L = np.sqrt(D_e * tau_e)             # diffusion length
    # light genearated current 
    J_1 = e*phi_0*L^2*alpha/(1-L^2*alpha^2) * (-alpha - 1/L *np.tanh(d/L)+alpha*np.exp(-alpha*d)/(np.cosh(d/L)))        
    # recombination current
    J_2 = e*n_0 *L/tau_e * np.tanh(d/L)*(np.exp(e*V/(m*k*T))-1)
    return J_1 - J_2

# get m and j_0 
## log method 
def log_shokley(j_sc, V, j):
    """using the log naturural of shockley equation to find the slope and intercept,  
        ln(J_sc - J) = ln(J_0) + (e/mkT)V 
    """
    fx =np.log(j_sc - j)
    slope, intercept = np.polyfit(V, fx,1) 
    j_0 = 10^(intercept)
    m = e/(slope*k*T)
    return j_0, m 

## non linear fitting
def non_liniear_fit_shokley (j_sc,j_0, m, V,j):
    """ shockley's equation, used to guest the ideality factor (m) and the saturation current (J_0), shot circuit current is gather from the data V for lowest J """
    shokley_equation= j_sc - j_0*(np.exp(e*V/(m*k*T))-1)
    initial_guest=log_shokley(j_sc,V,j)
    param_bounds = ( [0, 1.0], [np.inf, 5.0] )      # j > 0 & 1=<m=<5
    popt, pcov = curve_fit(shokley_equation,V,j,p0=initial_guest,bounds=param_bounds)
    errors = np.sqrt(np.diag(pcov))
    j_fit, m_fit = popt 
    return  j_fit, m_fit, errors