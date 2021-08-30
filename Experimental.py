import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import color
from matplotlib import image
from matplotlib.collections import EllipseCollection
from PIL import Image
from scipy.ndimage.measurements import center_of_mass

def Crop_Array(Image, height, width): # Crops array to desired heigh and width
    
    plt.imshow(Image, cmap='jet')
    plt.colorbar()
    center = np.round(plt.ginput(1))
    plt.close()
    
    height, width = height/2, width/2
    
    out = Image[int(center[0, 1] - height):int(center[0, 1] + height), 
                int(center[0, 0] - width):int(center[0, 0] + width)]

    return out

def Exp_SOP(R, L, H, D, back, crop, elres, el_map, int_map, fmt, centered):  # Generates polarization plot from Stokes intensities

    if fmt == 'Image':
        R, L, H, D = color.rgb2gray(image.imread(R)), color.rgb2gray(image.imread(L)), \
                     color.rgb2gray(image.imread(H)), color.rgb2gray(image.imread(D))
    elif fmt == 'Array':
        R, L, H, D = R, L, H, D
    
    
    R, L, H, D =  Crop_Array(R, crop, crop), Crop_Array(L, crop, crop), \
                  Crop_Array(H, crop, crop), Crop_Array(D, crop, crop)
                  
    R, L, H, D = R.astype(np.float64), L.astype(np.float64), H.astype(np.float64), \
                 D.astype(np.float64)  

    R, L, H, D = R - back*np.max(R+L), L - back*np.max(R+L), H - back*np.max(R+L), \
                 D - back*np.max(R+L),

    R[R<0] = 0
    L[L<0] = 0
    D[D<0] = 0
    H[H<0] = 0    

    S0 = R + L
    S1 = 2*H - S0
    S2 = 2*D - S0
    S3 = R - L    

    S0, S1, S2, S3 = S0.astype(np.float64), S1.astype(np.float64), S2.astype(np.float64), \
                 S3.astype(np.float64)   
                 
    Smag = np.sqrt(S1**2 + S2**2 + S3**2)
    S1, S2, S3 = S1/Smag, S2/Smag, S3/Smag
    S1, S2, S3 = np.nan_to_num(S1, 0), np.nan_to_num(S2, 0), np.nan_to_num(S3, 0)
    
    # S0 = S0/np.max(S0)
    # S1, S2, S3 = S1*S0, S2*S0, S3*S0
                 
    s0, s1, s2, s3 = resize(S0, (elres, elres), anti_aliasing=True), \
                 resize(S1, (elres, elres), anti_aliasing=True), \
                 resize(S2, (elres, elres), anti_aliasing=True) ,\
                 resize(S3, (elres, elres), anti_aliasing=True)

    Int = R + L
    Int = Int/np.max(Int)
    
    # smag = np.sqrt(s1**2 + s2**2 + s3**2)
    # s1, s2, s3 = s1/smag, s2/smag, s3/smag
    
    # s0 = s0/np.max(s0)
    # s1, s2, s3 = s1*s0, s2*s0, s3*s0
    
    L_hyp = np.hypot(s1, s2)
    h = np.sqrt((np.sqrt(s1**2 + s2**2 + s3**2) - L_hyp)/2)*np.shape(Int)[0]/elres
    w = np.sqrt((np.sqrt(s1**2 + s2**2 + s3**2) + L_hyp)/2)*np.shape(Int)[0]/elres
    Psi = -np.angle(s1 + 1j*s2)/2
    ha_temp=s3;
    for i in range(0,h.shape[0]):
        for j in range(0,h.shape[1]):
            if abs(ha_temp[i,j])<=0.05:
                ha_temp[i,j]=0
            else:
                ha_temp[i,j]=np.sign(s3[i,j])
    Ha = ha_temp
    
    x = np.linspace(0, elres, elres)*np.shape(Int)[0]/elres
    X, Y = np.meshgrid(x, x)
    XY = np.column_stack((X.ravel(), Y.ravel()))
    
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.imshow(Int, cmap=int_map)
    ec = EllipseCollection(w, h, np.rad2deg(Psi), units='xy', offsets=XY,
                           transOffset=ax.transData, cmap=el_map,  clim=[-1, 1],
                           facecolors='none', linewidth=3)
    ec.set_array(Ha.ravel())
    ax.add_collection(ec)
    ax.autoscale_view()
    plt.axis('off')
    # plt.show()

    return S0, S1, S2, S3 

def Sensor_SOP(R, L, H, D, back, crop, elres, el_map, int_map):  # Generates polarization plot from Stokes intensities
    
    plt.imshow(R + L, cmap='jet')
    plt.colorbar()
    center = np.round(plt.ginput(1))
    plt.close()
    
    R, L, H, D =  R[int(center[0,1] - crop): int(center[0,1] + crop), int(center[0,0] - crop): int(center[0,0] + crop)], \
                  L[int(center[0,1] - crop): int(center[0,1] + crop), int(center[0,0] - crop): int(center[0,0] + crop)], \
                  H[int(center[0,1] - crop): int(center[0,1] + crop), int(center[0,0] - crop): int(center[0,0] + crop)], \
                  D[int(center[0,1] - crop): int(center[0,1] + crop), int(center[0,0] - crop): int(center[0,0] + crop)]

                  
    R, L, H, D = R.astype(np.float64), L.astype(np.float64), H.astype(np.float64), \
                 D.astype(np.float64)  

    R, L, H, D = R - back*np.max(R+L), L - back*np.max(R+L), H - back*np.max(R+L), \
                 D - back*np.max(R+L),

    R[R<0] = 0
    L[L<0] = 0
    D[D<0] = 0
    H[H<0] = 0    

    S0 = R + L
    S1 = -2*H + S0
    S2 = 2*D - S0
    S3 = R - L    

    S0, S1, S2, S3 = S0.astype(np.float64), S1.astype(np.float64), S2.astype(np.float64), \
                 S3.astype(np.float64)   
                 
    Smag = np.sqrt(S1**2 + S2**2 + S3**2)
    S1, S2, S3 = S1/Smag, S2/Smag, S3/Smag
    S1, S2, S3 = np.nan_to_num(S1, 0), np.nan_to_num(S2, 0), np.nan_to_num(S3, 0)
    
    # S0 = S0/np.max(S0)
    # S1, S2, S3 = S1*S0, S2*S0, S3*S0
                 
    s0, s1, s2, s3 = resize(S0, (elres, elres), anti_aliasing=True), \
                 resize(S1, (elres, elres), anti_aliasing=True), \
                 resize(S2, (elres, elres), anti_aliasing=True) ,\
                 resize(S3, (elres, elres), anti_aliasing=True)

    Int = R + L
    Int = Int/np.max(Int)
    
    smag = np.sqrt(s1**2 + s2**2 + s3**2)
    s1, s2, s3 = s1/smag, s2/smag, s3/smag
    
    s0 = s0/np.max(s0)
    s1, s2, s3 = s1*s0, s2*s0, s3*s0
    
    L_hyp = np.hypot(s1, s2)
    h = np.sqrt((np.sqrt(s1**2 + s2**2 + s3**2) - L_hyp)/2)*np.shape(Int)[0]/elres
    w = np.sqrt((np.sqrt(s1**2 + s2**2 + s3**2) + L_hyp)/2)*np.shape(Int)[0]/elres
    Psi = -np.angle(s1 + 1j*s2)/2
    ha_temp=s3;
    for i in range(0,h.shape[0]):
        for j in range(0,h.shape[1]):
            if abs(ha_temp[i,j])<=0.27:
                ha_temp[i,j]=0
            else:
                ha_temp[i,j]=np.sign(s3[i,j])
    Ha = ha_temp
    
    x = np.linspace(0, elres, elres)*np.shape(Int)[0]/elres
    X, Y = np.meshgrid(x, x)
    XY = np.column_stack((X.ravel(), Y.ravel()))
    
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.imshow(Int, cmap=int_map)
    ec = EllipseCollection(w, h, np.rad2deg(Psi), units='xy', offsets=XY,
                           transOffset=ax.transData, cmap=el_map,  clim=[-1, 1],
                           facecolors='none', linewidth=2)
    ec.set_array(Ha.ravel())
    ax.add_collection(ec)
    ax.autoscale_view()
    plt.axis('off')
    # plt.show()

    return S0, S1, S2, S3    

def BinaryHologram(U, X, Y, gx, gy):

    A = np.arcsin(abs(U)/np.pi)
    P = np.angle(U)/np.pi
    G = gx*X + gy*Y
    
    H = 0.5 + 0.5*np.sign(np.cos(2*np.pi*G + np.pi*P) - np.cos(np.pi*A))
    
    return H

#########################################################
# DMD based wavefront analysis from Shack-Hartmanngrams #
#########################################################


def DetermineGradients(Ref_Image, Sample_Image, crop, dx, f, thresh, x_res, y_res):

    ref = Image.open(r"" + str(Ref_Image) + ".bmp")
    ref = ref.rotate(0)
    Reference = color.rgb2gray(np.asarray(ref))

    spot = Image.open(r"" + str(Sample_Image) + ".bmp")
    spot = spot.rotate(0)
    Spots = color.rgb2gray(np.asarray(spot))

    ImsizeR = np.shape(Reference - Spots)
    ImsizeS = np.shape(Spots)

    crop = crop

    Reference = Reference[int(ImsizeR[0]/2 - crop): int(ImsizeR[0]/2 + crop), int(ImsizeR[1]/2 - crop): int(ImsizeR[1]/2 + crop)]
    Spots = Spots[int(ImsizeS[0]/2 - crop): int(ImsizeS[0]/2 + crop), int(ImsizeS[1]/2 - crop): int(ImsizeS[1]/2 + crop)]

    x_cent, y_cent = np.zeros((1, 2*crop)), np.zeros((1, 2*crop))

    Spots_x, Spots_y = np.sum(Spots, axis=0), np.sum(Spots, axis=1)
    Spots_x, Spots_y = Spots_x/np.max(Spots_x), Spots_y/np.max(Spots_y)

    Reference_x, Reference_y = np.sum(Reference, axis=0), np.sum(Reference, axis=1)
    Reference_x, Reference_y = Reference_x/np.max(Reference_x), Reference_y/np.max(Reference_y)

    temp = np.zeros((1, 2 * crop))

    for idx in range(int(np.asarray(np.shape(Spots_x)))):
        if Reference_x[idx] > thresh:
            switch = 1
        else:
            switch = 0
            x_cent[0, int(np.max(temp) - np.min(temp))] = 1
            temp = np.zeros((1, 2 * crop))
        if switch == 1:
            temp[0, idx] = idx
        else:
            temp[0, idx] = 0

    for idx in range(int(np.asarray(np.shape(Spots_x)))):
        if Reference_y[idx] > thresh:
            switch = 1
        else:
            switch = 0
            y_cent[0, int(np.max(temp) - np.min(temp))] = 1
            temp = np.zeros((1, 2 * crop))
        if switch == 1:
            temp[0, idx] = idx
        else:
            temp[0, idx] = 0
        # plt.plot(temp[0, :])
        # plt.show()

    x_cent[0, 0] = 0
    y_cent[0, 0] = 0

    xc = np.where(x_cent[0, :] == 1)
    yc = np.where(y_cent[0, :] == 1)
    xc, yc = xc[0], yc[0]

    # plt.plot(x_cent[0, :])
    # plt.show()
    #
    # plt.plot(y_cent[0, :])
    # plt.show()

    Sp = np.zeros_like(Spots)

    I_A = 12

    PhiX, PhiY = np.zeros((x_res, y_res)), np.zeros((x_res, y_res))
    Ints = np.zeros((x_res, y_res))

    for idx in range(x_res):
        for jdx in range(y_res):
            Sp[yc[idx] - I_A: yc[idx] + I_A, xc[jdx] - I_A: xc[jdx] + I_A] = 1
            I_AreaR = Reference[yc[idx] - I_A: yc[idx] + I_A, xc[jdx] - I_A: xc[jdx] + I_A]
            I_AreaS = Spots[yc[idx] - I_A: yc[idx] + I_A, xc[jdx] - I_A: xc[jdx] + I_A]
            tempR = center_of_mass(I_AreaR)
            tempS = center_of_mass(I_AreaS)
            PhiX[idx, jdx], PhiY[idx, jdx] = (tempS[0] - tempR[0])*(dx/f), (tempS[1] - tempR[1])*(dx/f)
            Ints[idx, jdx] = np.sum(I_AreaR)

    return PhiX, PhiY, Ints
    
                 