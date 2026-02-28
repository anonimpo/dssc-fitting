import numpy as np 
import pandas as pd
from scipy import constants , optimize
import matplotlib.pyplot as plt 
import scienceplots
plt.style.use(['science','no-latex','ieee'])
#import warnings

e, kB = constants.e, constants.Boltzmann
phi0, d0, T0, alpha0 =1e16 ,8e-4 ,3e2, 5e3

def extended_shockley(V, m, n_0, tau_e, D_e, phi_0=phi0, d=d0, T=T0, alpha=alpha0):
    """ extended shockley equation is an ideal shockley equation derive from

    Args:
        V (float): voltage
        m (_float_): ideality factor constrains to 1 - 10
        n_0 (_float_): electron density in equilibrium
        tau_e (_float_): electron lifetime
        phi_0 (_float_, estimated): incident photon flux. Defaults to phi0.
        d (_float_, estimated): thickness. Defaults to d0.
        T (_float_, estimated): temperature. Defaults to T0.
        alpha (_type_, estimated): absorption coefficient. Defaults to alpha0.

    Returns:
        float: total current density
    """
    L = np.sqrt(D_e *tau_e)
    jsc = e*phi_0*L^2 *alpha/(1-L^2 *alpha^2) *(-alpha + 1/L*np.tanh(d/L) +alpha *np.exp(-alpha*d)/(np.cosh(d/L)))
    j_0 = e*n_0*L/tau_e*np.tanh(d/L)* (np.exp(e*V/(m*kB*T)) )
    return jsc-j_0

file = "./data/shakinah.xlsx"
data = [pd.read_excel(file,sheet,header=2)  for sheet in pd.ExcelFile(file).sheet_names]
jsc = data[0].iloc[data[0]["WE(1).Potential (V)"].abs().idxmin(), 2]
I,V =data[0]["mili A"], data[0]["WE(1).Potential (V)"]

initial_geuss=[]
low_lim, up_lim = [],[]
prams, covs = optimize.curve_fit(extended_shockley,V,I,p0=initial_geuss, bounds=(low_lim, up_lim))
