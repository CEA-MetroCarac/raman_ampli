"""
Study of Raman amplification when varying the thickness
of the layer of interest.
"""
import matplotlib.pyplot as plt
import numpy as np

from raman_ampli.layer import Layer
from raman_ampli.stack import Stack
from raman_ampli.simulator import SimYoon


def substrate_amplification(make_plots=True):
    # layers definition
    d_0 = 0
    d_1 = np.linspace(0.1, 80, 100)
    d_2 = 285
    d_3 = 1000

    excitation = 532
    raman_shift = 384

    # stack creation / graphene
    sup = Layer('INFO_Air', 'Sup', 'inf')
    layer_1 = Layer('indices_MoS2_1ML_BIS', 'MoS2', d_1)
    layer_2 = Layer('INFO_SiO2', 'BOX', d_2)
    layer_3 = Layer('INFO_Si_mesh', 'Si', d_3)
    sub = Layer('INFO_Si', 'Sub', 'inf')

    # stack creation (including air = superstrate)
    raman_stack = Stack()
    raman_stack.append(sup)
    raman_stack.append(layer_1)
    raman_stack.append(layer_2)
    raman_stack.append(layer_3)
    raman_stack.append(sub)

    # creation of layer of interest
    layer_interest = raman_stack[3]

    # creation of variable layer
    layer_var = raman_stack[1]
    xlabel = layer_var.label

    # simulator creation
    sim = SimYoon(raman_stack, layer_interest,
                  layer_var)  # Yoon et al. model

    intensity = sim.amplification_other_layer(wavelength=excitation,
                                                    shift=raman_shift)

    if make_plots:
        plt.figure('Thickness study')
        plt.title(str(xlabel) + '-dependent Raman signal')
        plt.yscale('log')
        plt.plot(d_1, intensity)
        plt.xlabel(str(xlabel) + ' thickness [nm]')
        plt.ylabel('Raman intensity [a.u.]')
        plt.show()
        return
    else:
        return d_1, intensity


if __name__ == '__main__':
    substrate_amplification(make_plots=True)
