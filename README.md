# StructuredLightPackage
Python code containing objects and functions to automate the simulation of structured light modes.

Simulations.py contains the functions for generating:
 - A real space Grid() coordinate arrays.
 - Scalar Laguerre-Gaussian (LG), Hermite-Gaussian (HG) and Besse-Gaussian (BG) Mode objects generated using the Grid() coordinates.
 - VectorMode() objects can be created using scalar Mode() objects in either the horizonta-vertical (HV) linear polarisation basis or
   the right-left (RL) circular polarisation basis (the object output is the set of 4 Stokes parameters). 
 - The SOP() function can be used to generate an ellipse plot from the Stokes parameters of a scalar or vector field. 
 - The Independent_VQF() and Dependent_VQF() functions determine the 'Vector Quality Factor' concurrence based, respectively basis independent and dependent, classical                non-separability measurement from the Stokes parameters and the orthogonally polarized fields respectively.
 -  The StokeFields() function calculates the two Stokes fields from the Stokes parameters.
 -  The FFT() and IFFT() functions compute the Fourier and inverse Fourier transforms of a given complex field using the discrete Fourier and inverse Fourier transform matrices.
 -  The Exact_Propagation() function propagates an arbitrary complex field using an fft method.
 -  The Turb_Gen() function generates a Kolmogorov turbulence screen using the fft method including additional subharmonic sampling.
 -  The Zernike() function generates the Zernike phase for a given set of indices.
 -  The FractionalOAM() function generates the field for a stable (minimum number of unique Gouy phases) configuration of non-integer OAM.

Experimental.py contains the functions for:
 - Crop_Array() crops a given array to the desired dimensions.
 - Exp_SOP() and Sensor_SOP() generate an ellipse plots from experimentally measured Stokes intensities.
 - BinaryHologram() generates binary holograms used for complex amplitude modulation through a DMD.
 - DetermineGradients() generates the local x and y gradients measured from Shack-Hartmann spot arrays.
