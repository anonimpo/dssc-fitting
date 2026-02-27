import numpy as np
import scipy as sp
from sklearn.metrics import r2_score
from scipy.constants import physical_constants as pc
e,k=pc['elementary charge'][0],pc['Boltzmann constant in eV/K'][0]

def shockley_eq(j_sc,j_0,V,m,T):
    ""
    
    re = j_sc -j_0(np.exp(e*V/{m*k*T})-1)
    return re

def j_total():
    " "
    param1 = e* psi_0* D_e* tau_e* alpha* (1 -D_e* tau_e* alpha^2 )^-1
    param2 = -alpha+ np.sqrt(D_e* tau_e)^-1 + np.tanh(d* np.sqrt(D_e* tau_e)^-1)+ alpha* np.exp(-alpha* d)* (d* (np.cosh(d* ()))^-1)^-1  
        
