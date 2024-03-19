import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical

from skimage.util import view_as_windows
import numpy as np
from numpy.fft import fft, ifft
import math

def to_ten(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()

def fm_sweep_generation(t_dur):
    f1=70000
    f2=15000
    Fs=500000
    t=np.arange(0,t_dur-1/Fs,1/Fs)
    #print(np.shape(t))
    A=1
    SLOPE=(f2-f1)/t_dur
    F=f1+SLOPE*t/2
    x=A*np.sin(2*np.pi*F*t)

    #Nfft=16384
    return t,x

def distance_based_ipi(distance,Fs):
    #calculate IPI based on distance
    if(distance>0 and distance<0.5):
        IPI=(5*1e-3)*Fs
    else:
        IPI=(5*1e-3+((100-5)/(3.5-0.5))*(distance-0.5)*1e-3)*Fs
    PPR=(1/(IPI/Fs))
    IPI=round(IPI)
    tau=int(round(96*1e-3*Fs))
    k=6.26
    return IPI,tau,k

def HRTF_interp_func(HRTFl,HRTFr,azm,elv,d_azm,d_elv):
    distance = np.power(azm*np.pi/180-d_azm*np.pi/180,2)+np.power(elv*np.pi/180-d_elv*np.pi/180,2)
    sigma = 2.5
    weights = np.exp(-distance/(2*np.power(sigma,2)))
    HRTFl_interp = np.matmul(weights.T,HRTFl)/np.sum(weights)
    HRTFr_interp = np.matmul(weights.T,HRTFr)/np.sum(weights)
    return HRTFl_interp,HRTFr_interp

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(azim,elev,r):
    z=r*np.sin(elev*np.pi/180)
    rcoselev=r*np.cos(elev*np.pi/180)
    x=rcoselev*np.cos(azim*np.pi/180)
    y=rcoselev*np.sin(azim*np.pi/180)
    return x,y,z


def get_angles_from_R(R):
    theta = math.atan2(-R[0,1],R[1,1])
    phi=math.asin(R[2,1])
    gamma=math.atan(-R[2,0]/R[2,2])
    return theta,phi,gamma

def get_R_from_angles(theta,phi,gamma):
    R_z_theta=[[np.cos(theta*np.pi/180),-np.sin(theta*np.pi/180),0],
               [np.sin(theta*np.pi/180),np.cos(theta*np.pi/180),0],
               [0,0,1]]
    R_y_gamma=[[np.cos(gamma*np.pi/180),0,np.sin(theta*np.pi/180)],
               [0,1,0],
               [-np.sin(gamma*np.pi/180),0,np.cos(gamma*np.pi/180)]]
    R_x_phi=[[1,0,0],
             [0,np.cos(phi*np.pi/180),-np.sin(phi*np.pi/180)],
             [0,np.sin(phi*np.pi/180),np.cos(phi*np.pi/180)]]

    R=np.matmul(R_z_theta,np.matmul(R_x_phi,R_y_gamma))
    return R

def imp_res(HRTF_interp):
    IR_lr=[]
    for k, hrtf in enumerate(HRTF_interp):
      asd=np.concatenate((np.zeros((1,41)),hrtf,np.zeros((1,644))), axis=1)
      Y1=np.concatenate((asd,np.zeros((1,1)),np.conjugate(np.flip(asd[:,1:],axis=1))), axis=1)
      IR=np.real(ifft(Y1))
      IR_bar = np.concatenate((IR[:,1025:],IR[:,1:1024]),axis=1)
      IR_lr.append(IR_bar)
    return IR_lr

def form_feature_vec(x_l,x_r,window,stride):
      x_l_win=view_as_windows(x_l,window,stride)
      x_r_win=view_as_windows(x_r,window,stride)
      x_win_concat=np.concatenate((x_l_win,x_r_win),axis=1)
      x_win_zero_mean=x_win_concat-(np.tile(x_win_concat.mean(axis=1),(2*window,1))).T
      x_win_std=(np.tile(np.sqrt(np.sum(x_win_concat**2,axis=1)),(2*window,1))).T
      x_win_norm=np.divide(x_win_zero_mean,x_win_std)
      x_win_norm[np.isinf(x_win_norm)]=0
      return x_win_norm
