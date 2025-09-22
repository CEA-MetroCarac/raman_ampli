"""
Raman amplification factors simulation

Adaptation to N-layer stacks
"""
from cmath import pi
import numpy as np

import raman_ampli.array_def as convert


# Fresnel interface coefficients
def t_fresnel(n_0, n_1):
    """
    Fresnel transmission coefficient between media 0 and 1.
    Parameters
    ----------
    n_0
    n_1

    Returns
    -------
    Transmission coefficient in amplitude.
    """
    return np.divide(2 * n_0, n_0 + n_1)


def r_fresnel(n_0, n_1):
    """
    Fresnel reflection coefficient between media 0 and 1.
    Parameters
    ----------
    n_0
    n_1

    Returns
    -------
    Reflection coefficient in amplitude.
    """
    return np.divide(n_0 - n_1, n_0 + n_1)


# Norm of wavevector at desired wavelength in layer of index n
def wavevector(n, wvl):
    """
    Wavevector of photon of wavelength 'wvl' and propagating in a medium of
    refractive index n (complex).
    Parameters
    ----------
    n
    wvl

    Returns
    -------
    Wavevector
    """
    return np.divide(2 * pi * n, wvl)


# Exponential phase factor due to multireflections
def phase(beta, thick):
    """
    Phase factor associated to the propagation of light in a layer of thickness
    'th'.
    Parameters
    ----------
    beta
    thick

    Returns
    -------
    Phase factor
    """
    return np.exp(-1j * beta * thick)


# Fabry-Perot r & t amplitude coefficients - stack of 2 interfaces
def r_fp(r_ij, r_jk, exp_shift):
    """
    Reflection coefficient (in amplitude) of in a 2-interface Fabry-Perot
    stack.
    Parameters
    ----------
    r_ij
    r_jk
    exp_shift

    Returns
    -------
    Reflection coefficient
    """
    return (r_ij + r_jk * exp_shift ** 2) / (1 + r_ij * r_jk * exp_shift ** 2)


def t_fp(t_ij, t_jk, r_ij, r_jk, exp_shift):
    """
    Transmission coefficient (in amplitude) of in a 2-interface Fabry-Perot
    stack.
    Parameters
    ----------
    t_ij
    t_jk
    r_ij
    r_jk
    exp_shift

    Returns
    -------
    Transmission coefficient
    """
    return (t_ij * t_jk * exp_shift) / (1 + r_ij * r_jk * exp_shift ** 2)


def thick_list(stack):
    """
    Get the list of thicknesses from any 'stack'.
    Parameters
    ----------
    stack

    Returns
    -------
    Array of thicknesses.
    """
    thick = []
    for layer in stack:
        thick_i = layer.thickness
        thick.append(thick_i)
    return thick


def n_list(stack, wvl):
    """
    Get the list of refractive indices from any 'stack'.
    Parameters
    ----------
    stack
    wvl

    Returns
    -------
    Array of refractive indices
    """
    # loop to get refractive index list (for each layer)
    n = []
    for layer in stack:
        n_i = layer.mat.index(wvl)
        n.append(n_i)
    return n


def r_list(stack, wvl):
    """
    Get the list of reflection coefficients from any 'stack'.
    Parameters
    ----------
    stack
    wvl

    Returns
    -------
    Array of reflection coefficients.
    """
    # r list (Fresnel calculation from stack, from top to bottom)
    r = []
    n = n_list(stack, wvl)
    for i in range(len(n) - 1):
        r_i = r_fresnel(n[i], n[i + 1])
        r.append(r_i)
    return r


def t_list(stack, wvl):
    """
    Get the list of transmission coefficients from any 'stack'.
    Parameters
    ----------
    stack
    wvl

    Returns
    -------
    Array of transmission coefficients.
    """
    t = []
    n = n_list(stack, wvl)
    for i in range(len(n) - 1):
        t_i = t_fresnel(n[i], n[i + 1])
        t.append(t_i)
    return t


def rt_fresnel(stack, wvl):
    """
    Recursive computation of reflection and transmission coefficients from a
    set of layers 'stack'.
    Parameters
    ----------
    stack: list
        array of layers (material + thickness) >> sub stack (top, bottom,
    reversed...)
    w: float
        wavelength

    Returns
    -------
    r_stack, t_stack: complex
        of input (sub) stack
        stack may be from layer 0 to i, i to N-1, i to 0... defined in Factor
        function
    """

    if len(stack) == 2:

        n_0 = stack[0].mat.index(wvl)
        n_1 = stack[1].mat.index(wvl)
        r_01 = r_fresnel(n_0, n_1)
        t_01 = t_fresnel(n_0, n_1)
        return r_01, t_01

    elif len(stack) > 2:

        thick = thick_list(stack)
        n = n_list(stack, wvl)
        r = r_list(stack, wvl)
        t = t_list(stack, wvl)

        r_int, t_int = r[0], t[0]

        thick_last = thick[1]
        n_last = n[1]
        beta = wavevector(n_last, wvl)
        exp_shift = phase(beta, thick_last)

        stack.pop(0)
        r_rec, t_rec = rt_fresnel(stack, wvl)

        r_stack = r_fp(r_int, r_rec, exp_shift)
        t_stack = t_fp(t_int, t_rec, r_int, r_rec, exp_shift)

        return r_stack, t_stack


def factor_other(stack, layer_int, wvl, mode):
    """
    Computes interference-enhanced amplification factors of a layer of interest.
    These factors are calculated for a stack where only one layer thickness is
    changed (e.g., changing oxide layer thickness and fixed silicon layer).
    Parameters
    ----------
    stack = array of layers (material + thickness) >> total stack
    layer_int = defined layer of interest (the one which Raman-scatters)
    w = wavelength
    mode = 'abs' or 'scat' (string)

    Returns
    -------
    List of lists ('macro-list')
    If the 'layer of interest' has only one thickness, then the macro-list is
    of length equal to 1.
    Each sub-list contains the abs. or scat. values for each x (depth) in th_i
    th_LOI = [th_0...th_i...th_p-1]
    """

    n = n_list(stack, wvl)

    # index and thickness extraction from layer of interest
    index_int = stack.index(layer_int)
    th_layerint = stack[index_int].thickness

    # stack above i [interest] > transmission from super. to i
    stack_1 = stack.copy()
    stack_up = stack_1[: index_int + 1]
    _, t_in = rt_fresnel(stack_up, wvl)

    # reversed stack above i [interest] > internal reflection inside i (up)
    stack_2 = stack.copy()
    stack_up_r = stack_2[: index_int + 1]
    stack_up_r.reverse()
    r_up, t_up = rt_fresnel(stack_up_r, wvl)

    # stack below i [interest] > internal reflection inside i (down)
    stack_3 = stack.copy()
    stack_bottom = stack_3[index_int:]
    r_bottom, _ = rt_fresnel(stack_bottom, wvl)

    # Discretization of the thickness of the layer of interest
    th_layerint_x = np.arange(start=0, stop=th_layerint, step=0.1)

    beta_i = wavevector(n[index_int], wvl)

    # Phase factors
    exp_shift = np.exp(-2j * np.multiply(beta_i, th_layerint))
    exp_shift_x = np.exp(-1j * beta_i * np.subtract(2 * th_layerint, th_layerint_x))
    exp_x = np.exp(-1j * np.multiply(beta_i, th_layerint_x))

    # r,t coefficients (vectors)
    t_in = t_in[np.newaxis].T
    t_up = t_up[np.newaxis].T
    r_bottom = r_bottom[np.newaxis].T
    r_up = r_up[np.newaxis].T

    if mode == 'abs':
        # Absorption factor

        num = np.multiply(t_in,
                          np.add(exp_x, np.multiply(r_bottom, exp_shift_x)))
        den = np.subtract(1,
                          np.multiply(r_up, np.multiply(r_bottom, exp_shift)))

        f_x = np.divide(num, den)

    elif mode == 'scat':
        # Scattering factor

        num = np.multiply(t_up,
                          np.add(exp_x, np.multiply(r_bottom, exp_shift_x)))
        den = np.subtract(1,
                          np.multiply(r_up, np.multiply(r_bottom, exp_shift)))
        f_x = np.divide(num, den)

    else:
        print(
            "Error, respect 'mode' parameter possible values ('abs' "
            "or 'scat')")

    return f_x


def factor_layerint(stack, layer_int, wvl, mode):
    """

    Parameters
    ----------
    stack = array of layers (material + thickness) >> total stack
    layer_int = defined layer of interest (the one which Raman-scatters)
    w = wavelength
    mode = 'abs' or 'scat' (string)

    Returns
    -------
    List of lists ('macro-list')
    If the 'layer of interest' has only one thickness, then the macro-list is
    of length equal to 1.
    Each sub-list contains the abs. or scat. values for each x (depth) in d_i
    d_int = [d_0...d_i...d_p-1]
    """

    n = n_list(stack, wvl)

    # index and thickness extraction from layer of interest
    index_int = stack.index(layer_int)
    th_layerint = stack[index_int].thickness

    # stack above i [interest] > transmission from super. to i
    stack_1 = stack.copy()
    stack_up = stack_1[: index_int + 1]
    _, t_in = rt_fresnel(stack_up, wvl)

    # reversed stack above i [interest] > internal reflection inside i (up)
    stack_2 = stack.copy()
    stack_up_r = stack_2[: index_int + 1]
    stack_up_r.reverse()
    r_up, t_up = rt_fresnel(stack_up_r, wvl)

    # stack below i [interest] > internal reflection inside i (down)
    stack_3 = stack.copy()
    stack_bottom = stack_3[index_int:]
    r_bottom, _ = rt_fresnel(stack_bottom, wvl)

    # Discretization of layer of interest thickness
    step_dis = 0.1
    max_num = int(1 + (th_layerint[-1] - 0) / step_dis)
    th_layerintx = convert.array_from_thickness(array=th_layerint,
                                                num_integ=max_num,
                                                step_size=step_dis)
    th_layerint = th_layerint[np.newaxis].T

    beta_i = wavevector(n[index_int], wvl)

    # Phase factors
    exp_shift = np.exp(-2j * beta_i * th_layerint)
    exp_shift_x = np.exp(-1j * beta_i * (2 * th_layerint - th_layerintx))
    exp_x = np.exp(-1j * beta_i * th_layerintx)

    if mode == 'abs':
        # Absorption factor
        f_x = t_in * (exp_x + r_bottom * exp_shift_x) / (
                1 - r_up * r_bottom * exp_shift)
        # F.append(f_x)
    elif mode == 'scat':
        # Scattering factor
        f_x = t_up * (exp_x + r_bottom * exp_shift_x) / (
                1 - r_up * r_bottom * exp_shift)
    else:
        print(
            "Error, respect 'mode' parameter possible values ('abs' "
            "or 'scat')")
    return f_x


def square_mod(x):
    '''
    Computes the square modulus of any complex number x.
    Parameters
    ----------
    x

    Returns
    -------
    The square modulus of x.
    '''
    return x.real ** 2 + x.imag ** 2


def integral(f_ab, f_sc):
    """

    Parameters
    ----------
    f_ab
    f_sc

    Returns
    -------
    List of integrated Raman intensities. If the 'layer of interest' contains
    only one thickness, the list has only one element
    """

    factor = np.multiply(f_ab, f_sc)
    factor_sq = square_mod(factor)
    factor_sq = convert.skip_nan(factor_sq)
    integ = np.trapz(factor_sq, dx=0.1)

    return integ  # integrated Raman intensity
