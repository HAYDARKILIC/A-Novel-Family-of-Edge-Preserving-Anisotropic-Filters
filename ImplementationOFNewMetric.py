# How to Implement NewMetric by Python
# Haydar KILIC, 2022
import os
import cv2
import numpy as np
os.system('cls||clear')
v1=np.array([[-1.],[-1.]])
v2=np.array([[-1],[0]])
v3=np.array([[-1],[1]])
v4=np.array([[0],[-1]])
v5=np.array([[0],[0]])
v6=np.array([[0],[1]])
v7=np.array([[1],[-1]])
v8=np.array([[1],[0]])
v9=np.array([[1],[1]])
vstruct=([v1,v4,v7],[v2, v5, v8], [v3, v6, v9])
img=cv2.imread('Image/Lena.jpg',cv2.IMREAD_GRAYSCALE)
assert not isinstance(img,type(None)), 'image not found'
I=cv2.normalize(src=img, dst=None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
mean=0
sigma=0.1
gaussian = np.random.normal(mean, sigma, (I.shape[0],I.shape[1]))
I_Noisy=I+gaussian
# FILTER COMPUTING STARTING HERE ---------------------------------------------------------------------------------------
beta=1
dt=0.05
iterno=3
# NewMetric Function Definiton -----------------------------------------------------------------------------------------
def NewMetric(I=None,beta=None,dt=None,iterno=None):
    b2 = beta**2
    m = I.shape[0]
    n = I.shape[1]
    Ipad = np.pad(I,(1,1),'minimum')
    Deltag = np.zeros((m,n))
    # Differential
    for k in range(0,iterno+1):
        # Central Derivation
        Ix = (np.insert(I,-1,I[:,-1],1)[:,1:] - np.insert(I,0,I[:,0],1)[:,:-1]) / 2
        Iy = (np.insert(I,-1,[I[-1,:]],0)[1:,:] - np.insert(I,0,[I[0,:]],0)[:-1,:])/2 # y central difference
        Ixx = np.insert(I,-1,I[:,-1],1)[:,1:] - 2*I + np.insert(I,0,I[:,0],1)[:,:-1]# xx central difference
        Iyy = np.insert(I,-1,[I[-1,:]],0)[1:,:] -2*I + np.insert(I,0,[I[0,:]],0)[:-1,:] # yy central difference
        aa = np.insert(I[1:,1:],-1,[I[1:,1:][-1,:]],0)
        bb = np.insert(I[:-1,:-1],0,[I[:-1,:-1][0,:]],0)
        cc = np.insert(I[:-1,1:],0,[I[:-1,1:][0,:]],0)
        dd = np.insert(I[1:,:-1],-1,[I[1:,:-1][-1,:]],0)
        Ixy = ( np.insert(aa,-1,aa[:,-1],1) - np.insert(bb,0,bb[:,0],1) - 
                np.insert(cc,-1,cc[:,-1],1) + np.insert(dd,0,dd[:,0],1) ) /4 # xy central difference
        # Other Derivatives
        g11 = 1 + b2 * Ix ** 2
        g12 = b2 * Ix * Iy
        g22 = 1 + b2 * Iy ** 2
        g11k1 = 2 * b2 * Ixx * Ix
        g12k1 = b2 * (Ixx * Iy + Ixy * Ix)
        g22k1 = 2 * b2 * Ixy * Iy
        g11k2 = 2 * b2 * Ixy * Ix
        g12k2 = b2 * (Ixy * Iy + Iyy * Ix)
        g22k2 = 2 * b2 * Iyy * Iy
        Z = Ix ** 2 + Iy ** 2  # Square of Euclidean Norm
        Zk1 = 2 * (Ix * Ixx + Iy * Ixy)
        Zk2 = 2 * (Ix * Ixy + Iy * Iyy)
        detg = 1 + b2 * Z
        c = b2 * Z / detg
        ck1 = (b2 / detg ** 2) * Zk1
        ck2 = (b2 / detg ** 2) * Zk2
        for i in range(0,m):
            for j in range(0,n):
                # Calculation of direction vector
                window = Ipad[i:i+3,j:j+3]
                # Find the maximum value in the selected window
                arg_max=abs(window-window[1,1])
                [row,col] = np.unravel_index(arg_max.argmax(), arg_max.shape)
                v=vstruct[row][col] # Best selection of direction vector
                # --------------------------------------------------------------
                g = np.array([[g11[i,j],g12[i,j]],[g12[i,j],g22[i,j]]])
                gk1 = np.array([[g11k1[i,j],g12k1[i,j]],[g12k1[i,j],g22k1[i,j]]])
                gk2 = np.array([[g11k2[i,j],g12k2[i,j]],[g12k2[i,j],g22k2[i,j]]])
                vK = g * v # Covariant components of vector
                ginv =1/detg[i,j]*np.array([[g22[i,j],-g12[i,j]],[-g12[i,j],g11[i,j]]])
                V=v.T*g*v
                K = 1 + c[i,j] * V
                S=-c[i,j]/K
                gammainv=ginv+S*v.T*v
                # ---------------------------------------------------------------
                # Gamma Derivation:
                gammak1=gk1+ck1[i,j]*vK.T*vK
                gammak2=gk2+ck2[i,j]*vK.T*vK
                # Connection Coefficients
                Kon111=1/2*gammainv[0,0]*gammak1[0,0]+1/2*gammainv[0,1]*(2*gammak1[0,1]-gammak2[0,0])
                Kon112=1/2*gammainv[0,0]*gammak2[0,0]+1/2*gammainv[0,1]*gammak1[1,1]
                Kon122=1/2*gammainv[0,0]*(2*gammak2[0,1]-gammak1[1,1])+1/2*gammainv[0,1]*gammak2[1,1]
                Kon1=[Kon111,Kon112,Kon112,Kon122]
                Kon211=1/2*gammainv[1,0]*gammak1[0,0]+1/2*gammainv[1,1]*(2*gammak1[0,1]-gammak2[0,0])
                Kon212=1/2*gammainv[1,0]*gammak2[0,0]+1/2*gammainv[1,1]*gammak1[1,1]
                Kon222=1/2*gammainv[1,0]*(2*gammak2[0,1]-gammak1[1,1])+1/2*gammainv[1,1]*gammak2[1,1]
                Kon2=[Kon211,Kon212,Kon212,Kon222]
                # Laplace-Beltrami Operator
                Deltag[i,j]=gammainv[0,0]*(Ixx[i,j]-Kon111*Ix[i,j]-Kon211*Iy[i,j])+\
                            2*gammainv[0,1]*(Ixy[i,j]-Kon112*Ix[i,j]-Kon212*Iy[i,j])+\
                            gammainv[1,1]*(Iyy[i,j]-Kon122*Ix[i,j]-Kon222*Iy[i,j])
        I=I+dt*Deltag
    return I
# End of New Metric Function Definition --------------------------------------------------------------------------------
# END OF THE FILTER COMPUTING ------------------------------------------------------------------------------------------
I_Result=NewMetric(I_Noisy,beta,dt,iterno)
cv2.imshow('Filtered Image',I_Result)
cv2.waitKey(0)






