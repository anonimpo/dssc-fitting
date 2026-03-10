import numpy as np 
from scipy import constants


# constant and fixed paramter from reference [diantoro 2019, IOP Conf. Series: Mater. Sci. Eng. 515 (2019) 012016]
e, kB = constants.e, constants.Boltzmann
phi_0, d, T, alpha =1e16 ,8e-4 ,3e2, 5e3

def area():
    """area from manuskrip"""
    return 0.2 

def parameters_value_from_reference():
    """initial guess for the parameters, from reference [diantoro 2019, IOP Conf. Series: Mater. Sci. Eng. 515 (2019) 012016]

    [e, kB, phi_0, d, T, alpha, m, n_0, tau_e, D_e]
    """
    # constants
    e, kB = constants.e, constants.Boltzmann
    # fixed or approximate 
    phi_0, d, T, alpha =1e16 ,8e-4 ,3e2, 5e3
    # to be fitted
    m, n_0, tau_e, D_e = 4.5, 1e14, 1e-2, 1e-4

    return  [e, kB, phi_0, d, T, alpha, m, n_0, tau_e, D_e]

def diffusion_model_shockley(V, m, n_0, tau_e, D_e):#, phi_0=phi_0, d=d, T=T, alpha=alpha):
    """ extended shockley equation is an shockley like's equation derive from maxwell equation?

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
    # Diffusion length L
    L = np.sqrt(D_e * tau_e)
    # Pre-factor for the first large bracket
    denom = 1 - (L**2 * alpha**2)
    term1_prefix = (e * phi_0 * L**2 * alpha) / denom
    # Contents of the square bracket []
    bracket = -alpha + (1/L * np.tanh(d/L)) + (alpha * np.exp(-alpha * d) / np.cosh(d/L))
    # short circuit current (density) , can be estimate from I with V ~ 0
    j_sc = term1_prefix * bracket    
    # sqrt(D_e/tau_e) is the diffusion velocity
    j_0 = e * n_0 * np.sqrt(D_e / tau_e) * np.tanh(d/L)
    # Equation 20: j = j_sc - j_0 * (exp(eV/mkT) - 1)
    j_total = j_sc - j_0 * (np.exp((e * V) / (m * kB * T)) - 1)
    
    return j_total

def single_diode_shockley(V,I, n, I_0, I_L, R_s, R_sh):
    """
    Single-diode model equation for solar cells.

    Args:
        V (float): Voltage
        I (float): Current
        n (float): Ideality factor
        I_0 (float): Diode saturation current
        I_L (float): Light-generated current
        R_s (float): Series resistance
        R_sh (float): Shunt resistance

    Returns:
        I_model (float )= modeled current
    """
    # Calculate the current using the single-diode model reference from [https://www.originlab.com/doc/Origin-Help/FitFunc-Script-Access]
    I_model = I_L - I_0 * (np.exp((V + I * R_s) / (n * kB * T)) - 1) - (V + I * R_s) / R_sh
    
    return I_model

def shockley_diode_residual(I, *args):
    """
    Residual function for the single-diode model.

    Args:
        I (float): Measured current
        *args: Parameters for the single-diode model (V, n, I_0, I_L, R_s, R_sh)

    Returns:
        float: Residual (difference between measured and modeled current)
    """
    return I - single_diode_shockley(*args)