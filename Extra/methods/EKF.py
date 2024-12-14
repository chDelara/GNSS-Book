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

def filter_pos2(x0, P, p_arr, dt, Q, R):
    dt = dt
    z_num = len(p_arr.flatten())
    kf = EKF(dim_x=40, dim_z=z_num)
    
    ### initial state values
    kf.x = x0
    
    # state transition matrix
    kf.F = np.zeros((40,40))
    
    ### state transition of x, y, z, b, dx, dy, dz, db
    kf.F[0,0] = 1.0 # (x) + dt
    kf.F[1,1] = 1.0 # (y) + dt
    kf.F[2,2] = 1.0 # (z) + dt
    kf.F[3,3] = 1.0 # (b) + dt
    kf.F[4,4] = 1.0 # vx
    kf.F[5,5] = 1.0 # vy
    kf.F[6,6] = 1.0 # vz
    kf.F[7,7] = 1.0 # db
    
    # np.fill_diagonal(kf.F[8:,8:], 1)
    
    kf.F[0,1] = dt # x + (dt)
    kf.F[1,2] = dt # y + (dt)
    kf.F[2,3] = dt # z + (dt)
    kf.F[3,4] = dt # b + (dt)
    
    ### measurement noise matrix
    kf.R = np.eye(z_num)
    
    if np.isscalar(R):
        kf.R *= np.square(R)
    else:
        kf.R *= np.square(R)
        np.fill_diagonal(kf.R, np.square(R.flatten()))
    
    ### process noise matrix
    ### for x, y, z, vx, vy, vz
    kf.Q[0:3,0:3] = np.eye(3) * np.square(0.5)
    kf.Q[4:7,4:7] = np.eye(3) * np.square(0)    
    
    ### for b, db
    q = Q_discrete_white_noise(dim=2, var=np.square(Q))
    
    kf.Q[3,3] = q[0,0]
    kf.Q[3,4] = q[0,1]
    kf.Q[7,7] = q[1,1]
    kf.Q[7,6] = q[1,0]
    
    ### state covariance matrix
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

def Jacobian_PPP(x0, H_arr, pos_list):
    # f1 = 1575.42 #MHz
    # f2 = 1227.6 #MHz
    # f5 = 1176. #MHz

    # l1 = c / (f1 * 1e6) ###L1 wavelength
    # l2 = c / (f2 * 1e6) ###L2 wavelength
    # l_if = l1*l2/((77*l2) - (60*l1))
    
    x0 = x0.flatten()
    x,y,z,b = x0[0], x0[1], x0[2], x0[3]
    init_x = np.array([x,y,z,b])
    
    J = np.zeros((64, 40))
    J_sub = G_mat(init_x,H_arr)
    
    for J_index, sv_num in enumerate(pos_list):
        J_row_p = np.insert(J_sub[J_index],4,np.zeros(36))
        J_row_cp = J_row_p
        J_row_cp[8+sv_num] = 1
        
        J[sv_num] = J_row_p
        J[sv_num+32] = J_row_cp
        
    return J

def Hx(x,H_arr):
    x = x.flatten()
    x,y,z,b = x[0], x[1], x[2], x[3]
    init_x = np.array([x,y,z,b])
    
    return est_p(init_x,H_arr)

def Hx_PPP(x, H_arr, pos_list):
    x = x.flatten()
    x,y,z,b = x[0], x[1], x[2], x[3]
    init_x = np.array([x,y,z,b])
    p = est_p(init_x,H_arr)
    
    hx = np.zeros((64,1))
    for index, sv_num in enumerate(pos_list):
        hx[sv_num] = p[index]
        hx[sv_num+32] = p[index] 
        
    return hx