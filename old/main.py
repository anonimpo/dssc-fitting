#!/bin/python
import sys
from math import cosh, exp, sqrt, tanh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import scipy.optimize
from scipy.constants import alpha, elementary_charge, physical_constants

plt.style.use(["science", "notebook", "grid"])

try:
    data = pd.read_excel(sys.argv[1], sheet_name=None)
    sheets = tuple(data.keys())
except FileNotFoundError:
    print(f"Error: File not found: {sys.argv[1]}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    sys.exit(1)


def set_float128():
    """
    Sets the default floating-point precision to float128.
    Modifies the global float, linspace, and array functions.
    """
    global float, linspace, array
    float = np.float128
    linspace = lambda start, stop, num: np.linspace(
        start, stop, num, dtype=np.float128
    )
    array = lambda data: np.array(data, dtype=np.float128)


# Constants
elementary_charge = physical_constants["elementary charge"][0]  # Coulombs
boltzmann_constant = physical_constants["Boltzmann constant"][0]  # J/K
n_0 = 1e11  # Carrier concentration (cm^-3)
tau_e = 1e-2  # Electron lifetime (s)
D_e = 1.1e-4  # Electron diffusion coefficient (cm^2/s)
psi_0 = 1e16  # Photon flux (photons/cm^2/s)


# Set temperature to 300 K if not provided
if len(sys.argv) < 4:
    T = 300  # Kelvin
else:
    try:
        T = float(sys.argv[2])
    except ValueError:
        print("Error: Invalid temperature value.  Using default 300 K.")
        T = 300


# Shockley's Equation
def shockley_equation(V, J_sc, J_0, m):
    """
    Calculates the current density using Shockley's diode equation.

    Args:
        V: Voltage (V)
        J_sc: Short-circuit current density (A/cm^2)
        J_0: Saturation current density (A/cm^2)
        m: Ideality factor (dimensionless)

    Returns:
        Current density (A/cm^2)
    """
    return J_sc - J_0 * (
        exp(elementary_charge * V / (m * boltzmann_constant * T)) - 1
    )


def find_m(sheet, m=2.5):
    if sheet not in data.key():
        print(f'sheet "{sheet}" not found in the excel file')
        return

    x_data = data[sheet].iloc[:2]
    y_data = data[sheet].iloc[:1]

    ppram, pcov = scipy.optimize.curve_fit(
        shockley_equation, x_data, y_data, p0=[1, 1, m]
    )
    return ppram[2]


# J_total
def J_total(d, m, V):
    """
    Calculates the total current density based on a more complex model.

    Args:
        d: Device thickness (cm)
        m: Ideality factor (dimensionless)
        V: Voltage (V)

    Returns:
        Total current density (A/cm^2)
    """

    part1 = (
        elementary_charge
        * psi_0
        * D_e
        * tau_e
        * alpha
        / (1 - D_e * tau_e * alpha**2)
    )

    part2 = (
        -alpha
        + 1 / (sqrt(D_e * tau_e)) * tanh(d / sqrt(D_e * tau_e))
        + alpha * exp(-alpha * d) / cosh(d / sqrt(D_e * tau_e))
    )

    part3 = (
        -elementary_charge
        * n_0
        * sqrt(D_e / tau_e)
        * tanh(d / sqrt(D_e * tau_e))
        * (exp(elementary_charge * V / (m * boltzmann_constant * T)) - 1)
    )

    return part1 * part2 + part3
