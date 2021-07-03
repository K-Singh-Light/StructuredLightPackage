import numpy as np
from numpy.random import randn
from numpy.fft import fft2


#########################################
# 2D discreet Fourier transforms        #
#########################################

# U - field to be transformed
# dx - real space pixel size
# f - focal length of Fourier lens

def FFT(U, dx, f, wavelength):

    if np.shape(U)[0] != np.shape(U)[1]:
        print('Please use a square matrix')

    H = np.shape(U)[0]
    x = np.linspace(-H / 2, H / 2, H) * dx
    dk = wavelength * f / (H * dx)
    k = np.linspace(-H / 2, H / 2, H) * dk
    k = k / (dk ** 2)

    k = np.reshape(k, [H, 1])
    x = np.reshape(x, [H, 1])

    FT = (np.exp(-1j * 2 * np.pi / H)) ** (np.multiply(np.transpose(k), x))
    A = np.dot(U, FT)
    out = np.dot(np.transpose(FT), A)

    return out


## Inverse Fourier Transform matrix

def IFFT(U, dx, f, wavelength):

    if np.shape(U)[0] != np.shape(U)[1]:
        print('Please use a square matrix')

    H = np.shape(U)[0]
    x = np.linspace(-H / 2, H / 2, H) * dx
    dk = wavelength * f / (H * dx)
    k = np.linspace(-H / 2, H / 2, H) * dk
    k = k / (dk ** 2)

    k = np.reshape(k, [H, 1])
    x = np.reshape(x, [H, 1])

    FT = (np.exp(-1j * 2 * np.pi / H)) ** (np.multiply(np.transpose(k), x))
    IFT = np.conj(FT)
    A = np.dot(U, np.transpose(IFT))
    out = np.dot(IFT, A)

    return out

## Free space propagation

def ExactPropagation(U, z, wavelength, dx):

    N = np.shape(U)
    N = N[0]
    df = 1 / (N*dx)
    f = np.linspace(-N/2, N/2-1, N)*df
    Fx, Fy = np.meshgrid(f, f)

    k = 2*np.pi/wavelength
    H = np.exp(-1j*k*z*np.sqrt(1-(wavelength**2)*(Fx**2 + Fy**2)))

    Uz = np.fft.fftshift(np.fft.fft2(U))
    Uz = np.fft.ifft2(np.fft.ifftshift(Uz*H))

    return Uz

## Generates Kolgmogorov turbulence screens using the FFT method - including subharmonic sampling

def Turb_Gen(N, M, D, Dr, dx):

    out = np.zeros((N, M))

    r0 = D/Dr
    dim = np.min([N, M])
    Delta = 1/(dx*dim)

    mesh_dim = np.arange(0, dim) - dim/2 - 1/2
    nX, nY = np.meshgrid(mesh_dim, mesh_dim)
    Grid = np.real(np.exp(-1j*np.pi*(nX + nY)))
    Grid = np.sign(Grid)
    rr = (nX*nX + nY*nY)*Delta**2

    P_Kol = 0.1517*Delta/r0**(5/6)*rr**(-11/12)

    f0 = (randn(dim, dim) + 1j*randn(dim, dim))*P_Kol/np.sqrt(2)
    f1 = Grid*fft2(f0)

    # subharmonic sampling

    ary = [-0.25, -0.25, -0.25, 0.125, -0.125, -0.125,
           0, 0, 0, 0, 0.125, 0.125, 0.125, 0.25, 0.25,
           0.25]
    bry = [-0.25, 0, 0.25, -0.125, 0, 0.125, -0.25,
           -0.125, 0.125, 0.25, -0.125, 0, 0.125, -0.25,
           0, 0.25]
    dary = [0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.25,
            0.125, 0.125, 0.25, 0.125, 0.125, 0.125, 0.25,
            0.25, 0.25]
    dbry = [0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.25,
            0.125, 0.125, 0.25, 0.125, 0.125, 0.125,
            0.25, 0.25, 0.25]

    ary, bry = np.asarray(ary), np.asarray(bry)
    dary, dbry = np.asarray(dary), np.asarray(dbry)

    ss = (ary**2 + bry**2)*Delta**2
    Ps_Kol = 0.1517*Delta/r0**(5/6)*ss**(-11/12)
    f0 = (randn(1, 16) + 1j*randn(1, 16))*Ps_Kol/np.sqrt(2)
    fn = f1

    for pp in range(16):
        eks = np.exp(1j*2*np.pi*(nX*ary[pp] + nY*bry[pp])/dim)
        fn = fn + f0[0, pp]*eks*dary[pp]*dbry[pp]

    out[int(M/2 - dim/2): int(M/2 + dim/2), int(N/2 - dim/2): int(N/2 + dim/2)] = np.real(fn)

    return out

import math
import numpy as np


## Generates Zernike phase functions on disk of radius rho_max

def Zernike(n, m, rho, PHI, rho_max):

    rho = rho/np.max(rho_max)
    R = np.zeros_like(rho)

    m1 = abs(m)

    if np.mod(n-m1, 2) == 1:
        R = np.zeros_like(rho)
    else:
        for k in range(0, int((n-m1)/2) + 1):
            R = R + (-1)**k * math.factorial(n - k) / (math.factorial(k) *
                                                       math.factorial((n+m1)/2-k) *
                                                       math.factorial((n-m1)/2-k)) * \
                rho**(n-2*k)

    if m >= 0:
        Z = np.sqrt(2 * (n + 1)) * R * np.cos(m * PHI)
    else:
        Z = np.sqrt(2 * (n + 1)) * R * np.sin(m * PHI)

    Z[rho > 1] = 0

    out = Z

    return out

