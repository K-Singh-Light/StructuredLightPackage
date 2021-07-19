import numpy as np
import scipy
from scipy import special
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import math
from numpy.random import randn
from numpy.fft import fft2

def Grid(H, V, dx):  # Generates a real space grid of H x V pixels size dx (in m)
    x = np.linspace(-H/2, H/2, H)*dx
    y = np.linspace(-V/2, V/2, V)*dx
    X, Y = np.meshgrid(x, y)

    return X, Y

def Cart2Pol(X, Y):
    R = np.sqrt(X**2 + Y**2)
    TH = np.arctan2(Y, X)
    
    return R, TH


class Mode(object):  # Generates a scalar mode LG, HG or BG (idx2 corresponds to radial wavevector comp. for BG)
    def __new__(self, X, Y, idx1, idx2, w0, mode):
        self.X = X
        self.Y = Y
        self.idx1 = idx1
        self.idx2 = idx2
        self.w0 = w0
        self.mode = mode

        if self.mode == 'LG':
                R, TH = np.sqrt(self.X**2 + self.Y**2), np.arctan2(self.Y, self.X)
                l_idx, p_idx = self.idx1, self.idx2
                w0 = self.w0

                out = np.sqrt((2 * math.factorial(p_idx)) / (math.pi * math.factorial(abs(l_idx) + p_idx))) * \
                      ((np.sqrt(2) * R) / w0) ** abs(l_idx) * \
                      np.exp(-(R / w0) ** 2) * \
                      np.exp(1j * l_idx * TH) * \
                      scipy.special.assoc_laguerre(2 * (R ** 2 / w0 ** 2), p_idx, abs(l_idx))

                out = out / np.max(out)

        if self.mode == 'HG':
                X, Y = self.X, self.Y
                n_idx, m_idx = self.idx1, self.idx2
                w0 = self.w0

                out = (1 / w0) * np.sqrt(
                    (2 ** (1 - n_idx - m_idx)) / (math.pi * math.factorial(n_idx) * math.factorial(m_idx))) * \
                      scipy.special.eval_hermite(n_idx, (np.sqrt(2) * X) / w0) * \
                      scipy.special.eval_hermite(m_idx, (np.sqrt(2) * Y) / w0) * \
                      np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)

                out = out / np.max(out)

        if self.mode == 'BG':
            R, TH = np.sqrt(self.X ** 2 + self.Y ** 2), np.arctan2(self.Y, self.X)
            l_idx, kr = self.idx1, self.idx2
            w0 = self.w0

            out = scipy.special.jv(abs(l_idx), kr * R) * \
                  np.exp(1j * l_idx * TH) * \
                  np.exp(-(R ** 2 / w0 ** 2))

            out = out / np.max(out)

        return out


class VectorMode(object):  # Generates a vector mode as a superposition of two scalar modes in either the RL or HV bases
    def __new__(self, mode1, mode2, basis):
        self.mode1 = mode1
        self.mode2 = mode2
        self.basis = basis
        self.S0 = []
        self.S1 = []
        self.S2 = []
        self.S3 = []
        self.VQF = []

        if basis == 'HV':  # Stokes calculation for HV basis
            UH, UV = self.mode1, self.mode2
            UD = (1/np.sqrt(2))*(UH + UV)
            UR = (1/np.sqrt(2))*(UH + 1j*UV)

            IH, IV = abs(UH*np.conj(UH)), abs(UV*np.conj(UV))
            TOT = IH + IV
            IH, IV, ID, IR = IH/np.max(TOT), IV/np.max(TOT), abs(UD*np.conj(UD))/np.max(TOT), abs(UR*np.conj(UR))/np.max(TOT)

            S0 = IH + IV
            S1 = IH - IV
            S2 = 2*ID - S0
            S3 = 2*IR - S0

        if basis == 'RL':  # Stokes calculation for RL basis
            UR, UL = self.mode1, self.mode2
            UD = (1 / np.sqrt(2)) * (UR - 1j*UL)
            UH = (1 / np.sqrt(2)) * (UR + UL)

            IR, IL = abs(UR * np.conj(UR)), abs(UL * np.conj(UL))
            TOT = IR + IL
            IR, IL, ID, IH = IR / np.max(TOT), IL / np.max(TOT), abs(UD * np.conj(UD)) / np.max(TOT), abs(
                UH * np.conj(UH)) / np.max(TOT)

            S0 = IR + IL
            S1 = 2*IH - S0
            S2 = 2*ID - S0
            S3 = IR - IL

        return S0, S1, S2, S3


def SOP(S0, S1, S2, S3, elres, dx):

    s0, s1, s2, s3 = resize(S0, (elres, elres), anti_aliasing=True), resize(S1, (elres, elres), anti_aliasing=True), \
                     resize(S2, (elres, elres), anti_aliasing=True), resize(S3, (elres, elres), anti_aliasing=True)

    L_hyp = np.hypot(s1, s2)
    h = np.sqrt((np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) - L_hyp) / 2) * dx  # Semi-minor axis
    w = np.sqrt((np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) + L_hyp) / 2) * dx  # Semi-major axis
    Psi = np.angle(s1 + 1j * s2) / 2  # Ellipse orientation angle
    Ha = np.sign(s3)  # Circular polarisation handedness (allows for inhomogeneous handedness)

    # Ellipses Plotting #

    x = np.linspace(-elres / 2, elres / 2, elres)
    X, Y = np.meshgrid(x, x)
    #
    XY = np.column_stack((X.ravel(), -Y.ravel())) * dx
    #
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    #
    ec = EllipseCollection(w, h, np.rad2deg(Psi), units='xy', offsets=XY,
                           transOffset=ax.transData, cmap='bwr', clim=[-1, 1], facecolors='none', lw=3)
    ec.set_array(Ha.ravel() * s0.ravel())
    #
    ax.add_collection(ec)
    ax.autoscale_view()
    c_bar = plt.colorbar(ec)
    plt.show()


def VQF(S0, S1, S2, S3):

    s0, s1, s2, s3 = np.sum(S0), np.sum(S1), np.sum(S2), np.sum(S3)

    VQF = np.real(np.sqrt(1 - (s1**2 + s2**2 + s3**2)/s0**2))

    return VQF

def StokesFields(S1, S2, S3):
    
    A12 = np.sqrt(S1**2 + S2**2)
    P12 = np.arctan2(S2, S1)
    
    A23 = np.sqrt(S2**2 + S3**2)
    P23 = np.arctan2(S3, S1)
    
    U12 = A12*np.exp(1j*P12)
    U23 = A23*np.exp(1j*P23)
    
    return U12, U23

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

