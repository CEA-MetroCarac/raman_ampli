"""
Study of Raman amplification when varying the thickness
of the layer of interest.
"""
import matplotlib.pyplot as plt
import numpy as np

from raman_ampli.layer import Layer
from raman_ampli.stack import Stack
from raman_ampli.simulator import SimYoon

if __name__ == '__main__':
    # layers definition
    d_0 = 0
    d_1 = np.linspace(0.1, 20, 100)
    d_11 = 10
    d_2 = 100

    excitation = 532
    raman_shift = 1586

    # stack creation / graphene
    sup = Layer('INFO_Air', 'Sup', 'inf')
    layer_0 = Layer('INFO_SiO2', 'Surf. Oxide', d_0)
    layer_1 = Layer('INFO_Graphene', 'Graphene', d_1)
    layer_11 = Layer('INFO_Si', 'SOI', d_11)
    layer_2 = Layer('INFO_SiO2', 'BOX', d_2)
    sub = Layer('INFO_Si', 'Sub', 'inf')

    # stack creation (including air = superstrate)
    raman_stack = Stack()
    raman_stack.append(sup)
    # raman_stack.append(layer_0)
    raman_stack.append(layer_1)
    # raman_stack.append(layer_11)
    raman_stack.append(layer_2)
    raman_stack.append(sub)

    # creation of layer of interest
    layer_interest = raman_stack[1]

    # creation of variable layer
    layer_var = raman_stack[1]
    xlabel = layer_var.label

    # simulator creation
    sim = SimYoon(raman_stack, layer_interest,
                  layer_var)  # Yoon et al. model

    intensity = sim.amplification_layer_of_interest(wavelength=excitation,
                                                    shift=raman_shift)

    plt.figure('Thickness study')
    plt.title(str(xlabel) + '-dependent Raman signal')

    plt.plot(d_1, intensity)

    plt.xlabel(str(xlabel) + ' thickness [nm]')
    plt.ylabel('Raman intensity [a.u.]')

    plt.show()
