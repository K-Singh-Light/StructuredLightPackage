# StructuredLightPackage
Python code containing objects and functions to automate the simulation of structured light modes.
Simulations.py contains the functions for generating a real space Grid() coordinate arrays,
scalar Laguerre-Gaussian (LG), Hermite-Gaussian (HG) and Besse-Gaussian (BG) Mode objects generated using the Grid() coordinates.
VectorMode() objects can be created using scalar Mode() objects in either the horizonta-vertical (HV) linear polarisation basis or
the right-left (RL) circular polarisation basis (the object output is the set of 4 Stokes parameters). The SOP() function can be
used to generate an ellipse plot from the Stokes parameters of a scalar or vector field. The VQF() function determines the
'Vector Quality Factor' concurrence based, basis independent, classical non-separability measurement from the Stokes parameters. 
Experimental.py code contains code to analyse experimental data.
