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

def filter_pos(x0, p_arr, Q, R):
    dt = 1.0
    z_num = len(p_arr.flatten())
    kf = EKF(dim_x=8, dim_z=z_num)
    
    kf.x = x0
    # state transition matrix
    kf.F = np.array([[1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    kf.R = np.eye(z_num) * np.square(R)
    kf.R[-1,-1] = 1e-9 * c # clock drift rate
    kf.R[-2,-2] = 1e-8 * c # clock drift
    
    kf.Q[0:6,0:6] = np.eye(6) * np.square(0.01)
    kf.Q[6:,6:] = Q_discrete_white_noise(dim=2, var=np.square(Q))    
    
    kf.P *= 50
    
    return kf

def Jacobian(x,H_arr):
    x = x.flatten()
    x,y,z,b = x[0], x[2], x[4], x[6]
    init_x = np.array([x,y,z,b])
    
    z_num = H_arr.shape[0]
    J = G_mat(init_x,H_arr)
    J = np.insert(J, [1,2,3,4], np.zeros(z_num)[np.newaxis].T, axis=1)
    return J

def Hx(x,H_arr):
    x = x.flatten()
    x,y,z,b = x[0], x[2], x[4], x[6]
    init_x = np.array([x,y,z,b])
    
    return est_p(init_x,H_arr)