"""
Class for exaltation simulation.
"""
import numpy as np

import raman_ampli.model as m


class SimYoon:
    '''
    Class defining two methods.
    - Method 'amplification_layer_of_interest' --> how Raman intensities are
    amplified when varying the thickness of the layer of interest
    - Method 'amplification_other_layer' --> how Raman intensities are
    amplified when varying the thickness of another layer
    '''

    def __init__(self, stack, layer_of_interest, variable_layer):
        self.stack = stack
        self.layer_of_interest = layer_of_interest
        self.variable_layer = variable_layer

    # test generalization N layers
    def amplification_layer_of_interest(self, wavelength, shift):
        '''
        Method defined to assess how Raman intensities are
        amplified when varying the thickness of the layer of interest
        Parameters
        ----------
        wavelength
        shift

        Returns
        -------

        '''
        # wavelength (all lengths in nanometers)
        wvl = wavelength

        # stack
        stack = self.stack

        # layer of interest
        loi = self.layer_of_interest

        f_ab = m.factor_layerint(stack, loi, wvl, 'abs')

        wvl_sc = np.divide(1, np.divide(1, wvl) - np.multiply(shift, 1e-7))
        f_sc = m.factor_layerint(stack, loi, wvl_sc, 'scat')

        intensity = m.integral(f_ab, f_sc)
        #intensity = intensity / np.max(intensity)

        return intensity

    def amplification_other_layer(self, wavelength, shift):
        '''
        Method defined to assess how Raman intensities are
        amplified when varying the thickness of another layer
        Parameters
        ----------
        wavelength
        shift

        Returns
        -------

        '''
        # wavelength (all lengths in nanometers)
        wvl = wavelength

        # stack
        stack = self.stack

        # layer of interest
        loi = self.layer_of_interest
        lvar = self.variable_layer

        f_ab = m.factor_other(stack, loi, wvl, 'abs')

        wvl_sc = np.divide(1, np.divide(1, wvl) - np.multiply(shift, 1e-7))
        f_sc = m.factor_other(stack, loi, wvl_sc, 'scat')

        intensity = m.integral(f_ab, f_sc)
        #intensity = intensity / np.max(intensity)

        return intensity
