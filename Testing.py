import matplotlib.pyplot as plt
import numpy as np
from Mode import Grid, Mode, VectorMode, SOP, VQF

# Code demonstrating use of the Structured Light package (all spatial units in m)

X, Y = Grid(1080, 1080, 10e-6)  # Grid(columns, rows, pixel size)

U1 = Mode(X, Y, 4, 2, 2e-3, 'LG')  # Mode(X, Y, l index, p index, focus waist, Laguerre-Gaussian mode)
U2 = Mode(X, Y, -4, 2, 2e-3, 'LG')

S0, S1, S2, S3 = VectorMode(U1, U2, 'RL')  # VectorMode(field1, field2, right-left circularly polarized basis)

SOP(S0, S1, S2, S3, 45, 100e-6)  # SOP(S0, S1, S2, S3, ellipse resolution, ellipse size)

VQF = VQF(S0, S1, S2, S3)  # VQF(S0, S1, S2, S3)

print(VQF)
