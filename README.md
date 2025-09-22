# Raman Amplification

## Installation

```
pip install git+https://github.com/CEA-MetroCarac/raman_ampli.git
```

## Execution

```
raman_ampli
```

then, select a **input_MODEL.xlsx** or similar file.
(Example of such a file is given [here](https://github.com/CEA-MetroCarac/raman_ampli/tree/main/raman_ampli/assets/input_MODEL.xlsx) to copy/paste into your project directory).

Or, from python scripting:

- using *.xlsx* files:

```
from raman_ampli.main import launcher

launcher(['..../input_MODEL1.xlsx', '..../input_MODEL2.xlsx', ...]) 
```

- without *.xlsx* file : refers to examples [here](https://github.com/CEA-MetroCarac/raman_ampli/tree/main/raman_ampli/examples)

## Acknowledgements
Part of this work, carried out on the CEA - Platform for Nanocharacterisation (PFNC), was supported by the “Recherche Technologique de Base” program of the French National Research Agency (ANR). Thanks to the contributors of this project:
Damien Monteil, PhD student generalized the physical model initially proposed for interference-enhanced Raman spectroscopy in graphene-on-oxide structures [1]. His work consisted in generalizing the computation of Raman amplification factors for stacks with as many layers as desired by the user.
Yann Mazel proposed and developed the initial architecture of the project, categorizing the data necessary for the simulation to work properly (optical index tables, layers, stacks...).
Finally, Patrick Quéméré carried out the final integration of this project into CEA-MetroCarac. Thanks to him for the improved robustness and readability he was able to bring to it with a view to future use by other contributors.

## Bibliography
[1] Yoon, Duhee, et al. "Interference effect on Raman spectrum of graphene on SiO 2/Si." Physical Review B—Condensed Matter and Materials Physics 80.12 (2009): 125422.
