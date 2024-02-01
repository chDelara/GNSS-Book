# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:54:49 2024

@author: ASTI
"""

import numpy as np
from scipy.constants import c
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
from .localization import G_mat, est_p

def filter_pos(x0, P, p_arr, dt, Q, R):
    dt = dt
    z_num = len(p_arr.flatten())
    kf = EKF(dim_x=8, dim_z=z_num)
    
    kf.x = x0
    # state transition matrix
    kf.F = np.array([[1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    
    kf.R = np.eye(z_num) * np.square(R)
    
    
    kf.Q[0:3,0:3] = np.eye(3) * np.square(2.0)
    kf.Q[4:-1,4:-1] = np.eye(3) * np.square(0)    
    
    q = Q_discrete_white_noise(dim=2, var=np.square(Q))
    kf.Q[3,3] = q[0,0]
    kf.Q[3,4] = q[0,1]
    kf.Q[-1,-1] = q[1,1]
    kf.Q[-1,-2] = q[1,0]
    
    if np.isscalar(P):
        kf.P *= P
    else:
        kf.P = P
    
    return kf

def Jacobian(x0,H_arr):
    x0 = x0.flatten()
    x,y,z,b = x0[0], x0[1], x0[2], x0[3]
    init_x = np.array([x,y,z,b])
    
    J = G_mat(init_x,H_arr)
    
    if len(x0) > 4:
        z_num = H_arr.shape[0]
        J = np.insert(J, 4, np.zeros((4,z_num)), axis=1)
    else:
        pass
    return J

def Hx(x,H_arr):
    x = x.flatten()
    x,y,z,b = x[0], x[1], x[2], x[3]
    init_x = np.array([x,y,z,b])
    
    return est_p(init_x,H_arr)