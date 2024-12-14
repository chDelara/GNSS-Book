# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:30:59 2024

@author: ASTI
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
from .sat_pos import calcSatBias, calcSatPos, precise_orbit
from .localization import ecef2enu, cart2ell, calc_az_el, rot_satpos, G_mat, est_p
from .iono_est import klobuchar
from .tropo_est import saastamoinen
from .EKF import filter_pos, filter_pos2, Jacobian, Jacobian_PPP, Hx, Hx_PPP
from scipy.constants import c
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import predict, update

f1 = 1575.42 #MHz
f2 = 1227.6 #MHz
f5 = 1176. #MHz

l1 = c / (f1 * 1e6) ###L1 wavelength
l2 = c / (f2 * 1e6) ###L2 wavelength
l5 = c / (f5 * 1e6) ###L5 wavelength

l_if = l1*l2/((77*l2) - (60*l1)) ###Ionospheric-free Combination wavelength
l_12 = c/((f1-f2) * 1e6) ###Wide Lane Combination wavelength
fl_12 = f1-f2

n_12 = c/((f1+f2) * 1e6) ###Narrow Lane Combination wavelength
fn_12 = f1+f2

def calc_rcvr_pos(rcvr_time,nav,group,x0,count):
    init_x = x0
        
    print(f"Epoch: {rcvr_time}")
    
    while True:
        H = np.empty((0,3))
        p_arr = np.empty((0,1))
        group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
        
        for sv in group.SV.unique():
            group_sub = group[group.SV == sv]
            p1 = group_sub.C1C.values
            p2 = group_sub.C2W.values
            pseudorange = (((f1**2) / ((f1**2) - (f2**2))) * p1) - (((f2**2) / ((f1**2) - (f2**2))) * p2)
            p = pseudorange[np.newaxis,:] + (calcSatBias(rcvr_time,nav,sv)*c) - init_x[-1]
            
            ###pseudoranges
            # pseudorange1 = group_sub.C1C
            # p1 = pseudorange1.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
            
            # pseudorange2 = group_sub.C2W
            # p2 = pseudorange2.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
            
            # ###carrier phase measurements
            # carrier1 = group_sub.L1C
            # cp1 = carrier1.values[:,np.newaxis] * c/(f1*1e6) + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
            
            # carrier2 = group_sub.L2W
            # cp2 = carrier2.values[:,np.newaxis] * c/(f2*1e6) + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
            
            # f_arr = np.vstack([p1,p2,cp1,cp2])
            
            # p, iono_delay, N1, N2 = dualF_eq(f_arr,f1,f2)
            # p = p[:,np.newaxis]
            
            sat_pos = calcSatPos(rcvr_time, p.flatten()[0], nav, sv)
            sat_pos = rot_satpos(sat_pos,p)
            
            if count == 0:
                H = np.vstack((H,sat_pos.T))    
                p_arr = np.append(p_arr,p,axis=0)
            
            else:
                azimuth, elevation = calc_az_el(ecef2enu(np.array(init_x[:-1])[np.newaxis,:],sat_pos.T))
                
                if elevation < 5.:
                    continue
                else:
                    H = np.vstack((H,sat_pos.T))    
                    p_arr = np.append(p_arr,p,axis=0)
        
        if len(H) < 4:
            raise Exception("Less than 4 satellites in view")
        else:
            pass
        
        x = init_x
        delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (-p_arr + est_p(x,H) - init_x[-1] + x[-1])
        x = x + delta_xb.flatten()
        
        counter = 0
        while ((np.abs(delta_xb) > 1e-8).all()) & (counter <= 100):
            delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (-p_arr + est_p(x,H) - init_x[-1] + x[-1])
            x = x + delta_xb.flatten()
            counter += 1
        
        if (np.abs(x - init_x) > 1e-8).all():
            init_x = x
            
        else:
            init_x = x
            break
    
    return init_x

def calc_rcvr_pos2(rcvr_time,nav,group,x0,count,freq='single',iono_corr=None):
    init_x = x0
        
    print(f"Epoch: {rcvr_time}")
    
    H = np.empty((0,3))
    p_arr = np.empty((0,1))
    try:
        group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
    except KeyError:
        group = group.dropna(subset=['C1C'])
        
    for sv in group.SV.unique():
        group_sub = group[group.SV == sv]
        nav_n = nav[nav['SV'] == sv]
        epoch_idx = abs(nav_n.toe - rcvr_time).idxmin()
        TGD = nav_n.loc[epoch_idx:epoch_idx,:].TGD.values
        
        pseudorange = 0
        
        if freq == 'single':
            pseudorange = group_sub.C1C.values
            carrier = group_sub.L1C.values * l1
            
        elif freq == 'dual':
            p1 = group_sub.C1C.values
            p2 = group_sub.C2W.values
            pseudorange = (((f1**2) / ((f1**2) - (f2**2))) * p1) - (((f2**2) / ((f1**2) - (f2**2))) * p2)
        
        p = pseudorange[np.newaxis,:] + ((calcSatBias(rcvr_time,nav,sv) - TGD)*c) - init_x[-1]
        
        ###pseudoranges
        # pseudorange1 = group_sub.C1C
        # p1 = pseudorange1.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # pseudorange2 = group_sub.C2W
        # p2 = pseudorange2.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # ###carrier phase measurements
        # carrier1 = group_sub.L1C
        # cp1 = carrier1.values[:,np.newaxis] * c/(f1*1e6) + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # carrier2 = group_sub.L2W
        # cp2 = carrier2.values[:,np.newaxis] * c/(f2*1e6) + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # f_arr = np.vstack([p1,p2,cp1,cp2])
        
        # p, iono_delay, N1, N2 = dualF_eq(f_arr,f1,f2)
        # p = p[:,np.newaxis]
        
        sat_pos = calcSatPos(rcvr_time, p.flatten()[0], nav, sv)
        sat_pos = rot_satpos(sat_pos,p)
        
        if count == 0:
            H = np.vstack((H,sat_pos.T))  
            p_arr = np.append(p_arr,p,axis=0)
        
        else:
            azimuth, elevation = calc_az_el(ecef2enu(np.array(init_x[:-1])[np.newaxis,:],sat_pos.T))
            
            if freq == 'single':
                iono_l1 = klobuchar(iono_corr, cart2ell(init_x[:-1]), azimuth, elevation, rcvr_time)*c
                if iono_corr != None:
                    p = p - iono_l1
                else:
                    pass
            
            elif freq == 'dual':
                pass
            
            if elevation < 15.:
                continue
            else:
                H = np.vstack((H,sat_pos.T))    
                p_arr = np.append(p_arr,p,axis=0)
    
    if len(H) < 4:
        raise Exception("Less than 4 satellites in view")
    else:
        pass
    
    x = init_x
    delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (p_arr - est_p(x,H) + init_x[-1] - x[-1])
    x = x + delta_xb.flatten()
    
    counter = 0
    while (~(np.abs(delta_xb) < 1e-3).all()) & (counter <= 100):
        delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (p_arr - est_p(x,H) + init_x[-1] - x[-1])
        x = x + delta_xb.flatten()
        
        counter += 1
    
    ### Computation of statistics
    residual = (p_arr - est_p(x,H) + init_x[-1] - x[-1]) - (G_mat(x,H)@delta_xb) 
    
    
    
    return x, residual

def calc_rcvr_pos3(rcvr_time,nav,group,x0,P,R_std,Q_std,count,freq='single',iono_corr=None):
    print(f"Epoch: {rcvr_time}")
    
    init_x = x0
    x,y,z,b = init_x.flatten()[:4]
    
    H = np.empty((0,3))
    p_arr = np.empty((0,1))
    
    try:
        group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
    except KeyError:
        group = group.dropna(subset=['C1C'])
        
    for sv in group.SV.unique():
        group_sub = group[group.SV == sv]
        nav_n = nav[nav['SV'] == sv]
        try:
            epoch_idx = abs(nav_n.toe - rcvr_time).idxmin()
        except ValueError:
            continue
        TGD = nav_n.loc[epoch_idx:epoch_idx,:].TGD.values
        
        
        pseudorange = 0
        if freq == 'single':
            pseudorange = group_sub.C1C.values
            carrier = group_sub.L1C.values * l1
            
        elif freq == 'dual':
            p1 = group_sub.C1C.values
            p2 = group_sub.C2W.values
            pseudorange = (((f1**2) / ((f1**2) - (f2**2))) * p1) - (((f2**2) / ((f1**2) - (f2**2))) * p2)
        
        p = pseudorange[np.newaxis,:] + ((calcSatBias(rcvr_time,nav,sv) - TGD)*c) - b
        
        ###pseudoranges
        # pseudorange1 = group_sub.C1C
        # p1 = pseudorange1.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # pseudorange2 = group_sub.C2W
        # p2 = pseudorange2.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # ###carrier phase measurements
        # carrier1 = group_sub.L1C
        # cp1 = carrier1.values[:,np.newaxis] * c/(f1*1e6) + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # carrier2 = group_sub.L2W
        # cp2 = carrier2.values[:,np.newaxis] * c/(f2*1e6) + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
        
        # f_arr = np.vstack([p1,p2,cp1,cp2])
        
        # p, iono_delay, N1, N2 = dualF_eq(f_arr,f1,f2)
        # p = p[:,np.newaxis]
        
        sat_pos = calcSatPos(rcvr_time, p.flatten()[0], nav, sv)
        sat_pos = rot_satpos(sat_pos,p)
        
        if count == 0:
            H = np.vstack((H,sat_pos.T))  
            p_arr = np.append(p_arr,p,axis=0)
        
        else:
            azimuth, elevation = calc_az_el(ecef2enu(np.array([x,y,z])[np.newaxis,:],sat_pos.T))
            
            if freq == 'single':
                if iono_corr is not None:
                    iono_l1 = klobuchar(iono_corr, cart2ell(np.array([x,y,z])), azimuth, elevation, rcvr_time)*c
                    p = p - iono_l1
                else:
                    pass
            
            elif freq == 'dual':
                pass
            
            if elevation < 15.:
                continue
            else:
                height = cart2ell(np.array([[x,y,z]])).flatten()[-1]
                Tr = saastamoinen(height,elevation[0])
                H = np.vstack((H,sat_pos.T))    
                p_arr = np.append(p_arr,p - Tr,axis=0)
    
    if len(H) < 4:
        raise Exception("Less than 4 satellites in view")
    else:
        pass
    
    
    x_n = init_x
    dt = 0.0 # GPS data frequency 1 data/second
    
    kf = filter_pos(x_n, P, p_arr, dt, Q_std, R_std)
    
    num_iter = 1
    
    for iteration in range(num_iter):
        kf.predict()
        kf.update(z=p_arr,HJacobian=Jacobian,Hx=Hx,args=H,hx_args=H)
        
        # eps = np.dot(np.dot(kf.y.T,np.linalg.inv(kf.S)),kf.y)
        # print("eps: ",eps)
        
        # xk, yk, zk, bk = kf.x[0],kf.x[1],kf.x[2],kf.x[3]
        # print(cart2ell(np.array([xk,yk,zk])).T,bk)
        
    return kf.x, kf.P, kf.log_likelihood
    # return kf

def calc_rcvr_pos4(rcvr_time,nav,group,x0,P,R_std,count,freq='single',iono_corr=None):
    print(f"Epoch: {rcvr_time}")
    
    init_x = x0
    x0,y0,z0,b0 = init_x.flatten()[:4]
    
    try:
        group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
    except KeyError:
        group = group.dropna(subset=['C1C'])
        
    converge = False
    while not converge:
        H = np.empty((0,3))
        p_arr = np.empty((0,1))
        el_list = []
        
        for sv in group.SV.unique():
            group_sub = group[group.SV == sv]
            nav_n = nav[nav['SV'] == sv]
            epoch_idx = abs(nav_n.toe - rcvr_time).idxmin()
            TGD = nav_n.loc[epoch_idx:epoch_idx,:].TGD.values
            
            pseudorange = 0
            
            if freq == 'single':
                pseudorange = group_sub.C1C.values
                carrier = group_sub.L1C.values * l1
                
            elif freq == 'dual':
                p1 = group_sub.C1C.values
                p2 = group_sub.C2W.values
                pseudorange = (((f1**2) / ((f1**2) - (f2**2))) * p1) - (((f2**2) / ((f1**2) - (f2**2))) * p2)
            
            p = pseudorange[np.newaxis,:] + ((calcSatBias(rcvr_time,nav,sv) - TGD)*c) - b0
            
            
            sat_pos = calcSatPos(rcvr_time, p.flatten()[0], nav, sv)
            sat_pos = rot_satpos(sat_pos,p)
            
            if count == 0:
                H = np.vstack((H,sat_pos.T))  
                p_arr = np.append(p_arr,p,axis=0)
                
            else:
                azimuth, elevation = calc_az_el(ecef2enu(np.array([x0,y0,z0])[np.newaxis,:],sat_pos.T))
                el_list.append(elevation)
                
                if freq == 'single':
                    if iono_corr is not None:
                        iono_l1 = klobuchar(iono_corr, cart2ell(np.array([x0,y0,z0])), azimuth, elevation, rcvr_time)*c
                        p = p - iono_l1
                    else:
                        pass
                
                elif freq == 'dual':
                    pass
                
                if elevation < 15.0:
                    continue
                else:
                    H = np.vstack((H,sat_pos.T))    
                    p_arr = np.append(p_arr,p,axis=0)
        
        if len(H) < 4:
            raise Exception("Less than 4 satellites in view")
        else:
            pass
        
        ### Generate a covariance matrix
        if np.isscalar(P):
            P = np.eye(len(p_arr)) * P
        else:
            pass
        
        ### Generate a Measurement noise matrix
        R = np.eye(len(p_arr)) * np.square(R_std)
        
        x_n = init_x
        delta_x = np.ones_like(x_n)
        
        while ~np.all(np.abs(delta_x) < 1e-3):
            # update step
            residual = p_arr - est_p(x_n.flatten(),H) + b0 - x_n.flatten()[-1]
            
            J = Jacobian(x_n,H)
            delta_x = np.dot( np.dot(la.inv(np.dot(J.T,J)), J.T), residual)
            x_n += delta_x
        
        x0,y0,z0,b0 = x_n.flatten()
        
        if np.all(np.abs(delta_x) < 1e-6):
            converge = True
        else:
            continue
    
    return x_n

def est_pIF(prange_arr,f1,f2):
    
    g_mat = np.array([[1,1],
                      [1,(f1**2)/(f2**2)]])
    
    pIF = la.inv(g_mat.T @ g_mat) @ g_mat.T @ prange_arr
    
    return pIF

def dualF_eq(f_arr,f1,f2):
    
    g_mat = np.array([[1,1,0,0],
                      [1,(f1**2)/(f2**2),0,0],
                      [1,-1,1,0],
                      [1,-(f1**2)/(f2**2),0,1]])
    
    pIF = la.inv(g_mat.T @ g_mat) @ g_mat.T @ f_arr
    
    return pIF

    
def calc_cp_pos(rcvr_time,nav,group,x0,N1_dict,N2_dict,N1_list,N2_list,count):
    init_x = x0
    
    print(f"Epoch: {rcvr_time}")
    
    while True:
        H = np.empty((0,3))
        p_arr = np.empty((0,1))
        group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
        
        for sv in group.SV.unique():
            group_sub = group[group.SV == sv]
            
            pseudorange = group_sub.C1C
            p = pseudorange.values[:,np.newaxis]
            
            ### Carrier Phases Measurements
            carrier1 = group_sub.L1C
            cp1 = carrier1.values[:,np.newaxis]
            
            carrier2 = group_sub.L2W
            cp2 = carrier2.values[:,np.newaxis]
            
            NL12 = cp1 - cp2 - (p / l_12)
            
            N1 = (((l2/l1)-1)**-1) * ( ((l2/l1)*NL12) - cp1 + ((l2/l1)*cp2))
            N2 = N1 - NL12
            
            if count == 0:
                N1 = N1[0][0]
                N2 = N2[0][0]
                N1_dict[sv] = N1
                N2_dict[sv] = N2
                
            elif (count <= 100) & ~(count == 0):
                sv_index = list(N1_dict.keys()).index(sv)

                N1 = np.around(np.mean(N1_list,axis=0)[sv_index])
                N2 = np.around(np.mean(N2_list,axis=0)[sv_index])
                N1_dict[sv] = N1
                N2_dict[sv] = N2
                
            else:
                sv_index = list(N1_dict.keys()).index(sv)
                # N1 = np.around(np.mean(N1_list,axis=0)).astype(int)[sv_index]
                # N2 = np.around(np.mean(N2_list,axis=0)).astype(int)[sv_index]
                N1 = np.around(np.mean(N1_list[:],axis=0)[sv_index])
                N2 = np.around(np.mean(N2_list[:],axis=0)[sv_index])
                N1_dict[sv] = N1
                N2_dict[sv] = N2
            
            
            p = p.flatten()[0]
            cp11 = (cp1 - N1)*l1 + (calcSatBias(rcvr_time,nav,sv)*c) - init_x[-1]
            cp22 = (cp2 - N2)*l2 + (calcSatBias(rcvr_time,nav,sv)*c) - init_x[-1]
            
            iono_l1 = ((f2**2)/(f1**2 - f2**2)) * ((cp11-(N1*l1)) - (cp22-(N2*l2))) ### ALready
            
            # cp = cp11 - iono_l1
            # cp = cp22 - ((f1**2/f2**2)*iono_l1)
            
            cp = ((((f1**2) / ((f1**2) - (f2**2))) * cp11) - (((f2**2) / ((f1**2) - (f2**2))) * cp22))
            
            p = p + (calcSatBias(rcvr_time,nav,sv)*c) - init_x[-1] - iono_l1
            sat_pos = calcSatPos(rcvr_time, p[0], nav, sv)
            sat_pos = rot_satpos(sat_pos,p)
            
            if count == 0:
                H = np.vstack((H,sat_pos.T))    
                p_arr = np.append(p_arr,cp,axis=0)
            
            else:
                azimuth, elevation = calc_az_el(ecef2enu(np.array(init_x[:-1])[np.newaxis,:],sat_pos.T))
                
                if elevation < 5.:
                    continue
                else:
                    H = np.vstack((H,sat_pos.T))    
                    p_arr = np.append(p_arr,cp,axis=0)
        
        if len(H) < 4:
            raise Exception("Less than 4 satellites in view")
        else:
            pass
        
        x = init_x
        delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (p_arr - est_p(x,H) + init_x[-1] - x[-1])
        x = x + delta_xb.flatten()
        
        counter = 0
        while ((np.abs(delta_xb) > 1e-3).all()) & (counter <= 100):
            delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (p_arr - est_p(x,H) + init_x[-1] - x[-1])
            x = x + delta_xb.flatten()
            counter += 1
        
        if (np.abs(x - init_x) > 1e-8).all():
            init_x = x
            
        else:
            init_x = x
            break
    
    return init_x

def calc_cp_pos2(rcvr_time, nav, sp3, group, x0, P, R_std, Q_std, N1_dict, count):
    
    init_x = x0
    x,y,z,b = init_x.flatten()[:4]
    
    print(f"Epoch: {rcvr_time}")
    
    ### initialize matrices to store satellite positions, pseudorange
    ### and carrier phase measurements
    H = np.empty((0,3))
    p_arr = np.zeros((32,1))
    cp_arr = np.zeros((32,1))
    elev_arr = np.zeros((32,1))
    
    ### drop records with nan C1C, C2W, L1C, and L2W values 
    group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
    
    ### store the order of the SV numbers
    sv_arr = []
    
    for sv in group.SV.unique():
        ### convert SV number into python index for arrangement of y (measurement) and G/J matrix
        sv_num = int(sv[1:]) - 1
        
        group_sub = group[group.SV == sv]
        
        ### satellite clock bias
        sat_bias = calcSatBias(rcvr_time,nav,sv)*c
        
        ### Pseudorange measurements
        pseudorange_l1 = group_sub.C1C
        pseudorange_l2 = group_sub.C2W
        p1 = pseudorange_l1.values[:,np.newaxis] + sat_bias - b
        p2 = pseudorange_l2.values[:,np.newaxis] + sat_bias - b
        
        ### Carrier Phases Measurements
        carrier_l1 = group_sub.L1C
        carrier_l2 = group_sub.L2W
        cp1 = (carrier_l1.values[:,np.newaxis] * l1) + sat_bias - b
        cp2 = (carrier_l2.values[:,np.newaxis] * l2) + sat_bias - b
        
        ### ionospheric-free Pseudorange and Carrier-phase measurements
        p_if = (((f1**2) / ((f1**2) - (f2**2))) * p1) - (((f2**2) / ((f1**2) - (f2**2))) * p2)
        cp_if = ((((f1**2) / ((f1**2) - (f2**2))) * cp1) - (((f2**2) / ((f1**2) - (f2**2))) * cp2))
        
        ### Wide-lane integer ambiguity
        # NL12 = (cp1 - cp2 - (p1 / l_12)).flatten()[0]
        # N1 = (((l2/l1)-1)**-1) * ( ((l2/l1)*NL12) - cp1 + ((l2/l1)*cp2))
        # N2 = N1 - NL12
        # N_if = N1 - ((60/77) * N2)
        
        ### ionospheric-free Pseudorange and Carrier-phase measurements
        # N_if = N1_dict[sv_num][0]
        # p_if = p_if + (calcSatBias(rcvr_time,nav,sv)*c) - b
        # cp_if = cp_if + (calcSatBias(rcvr_time,nav,sv)*c) - b
        
        if not isinstance(sp3, pd.DataFrame):
            sat_pos = calcSatPos(rcvr_time, p_if.flatten()[0], nav, sv)
            sat_pos = rot_satpos(sat_pos,p_if)

        else:
            sat_pos = precise_orbit(group_sub.epoch.values[0], p_if, sv, sp3)
            sat_pos = rot_satpos(sat_pos,p_if)
        
        if count == 0:
            H = np.vstack((H,sat_pos.T))
            p_arr[sv_num] = p_if
            cp_arr[sv_num] = cp_if
            sv_arr.append(sv_num)
        
        else:
            azimuth, elevation = calc_az_el(ecef2enu(np.array([x,y,z])[np.newaxis,:],sat_pos.T))
            
            if elevation < 15.:
                continue
            
            else:
                H = np.vstack((H,sat_pos.T))    
                p_arr[sv_num] = p_if
                cp_arr[sv_num] = cp_if
                elev_arr[sv_num] = 0.003 + (0.003/np.sin(np.radians(elevation[0])))
                sv_arr.append(sv_num)
    
    if len(H) < 4:
        raise Exception("Less than 4 satellites in view")
    else:
        pass
        
    x_n = init_x
    dt = 0.0 # GPS data frequency 1 data/second
    y_arr = np.vstack([p_arr,cp_arr])
    elev_arr = np.vstack([elev_arr,elev_arr])
    pos_list = np.where(p_arr != 0)[0].tolist()
    
    if count == 0:
        pass
    else:
        R_empt_std = R_std
        R_std = elev_arr
        R_std = np.where(R_std == 0, np.square(R_empt_std), np.square(R_std))
    
    kf = filter_pos2(x_n, P, y_arr, dt, Q_std, R_std)
    
    for iteration in range(1):
        kf.predict()
        kf.update(z=y_arr,HJacobian=Jacobian_PPP,Hx=Hx_PPP,args=(H,pos_list),hx_args=(H,pos_list))
        
    
    sv_list = np.where(kf.x[8:] != 0)[0].tolist()
    
    for sv in sv_list:
        # if N1_dict[sv][1] - kf.x[8+sv][0] > 5:
        N1_dict[sv][0] = rcvr_time
        N1_dict[sv][1] = kf.x[8+sv][0]/l_if
            
    
    return kf.x, kf.P, kf.log_likelihood, kf.y

def calc_cp_pos3(rcvr_time, nav, sp3, group, x0, P, R_std, Q_std, N1_dict, count):
    
    init_x = x0
    x,y,z,b = init_x.flatten()[:4]
    
    print(f"Epoch: {rcvr_time}")
    
    ### initialize matrices to store satellite positions, pseudorange
    ### and carrier phase measurements
    H = np.empty((0,3))
    p_arr = np.zeros((32,1))
    cp_arr = np.zeros((32,1))
    elev_arr = np.zeros((64,1))
    
    ### drop records with nan C1C, C2W, L1C, and L2W values 
    group = group.dropna(subset=['C1C','C2W','L1C','L2W'])
    
    ### store the order of the SV numbers
    sv_arr = []
    
    for sv in group.SV.unique():
        ### convert SV number into python index for arrangement of y (measurement) and G/J matrix
        sv_num = int(sv[1:]) - 1
        
        group_sub = group[group.SV == sv]
        
        ### satellite clock bias
        sat_bias = calcSatBias(rcvr_time,nav,sv)*c
        
        ### Pseudorange measurements
        pseudorange_l1 = group_sub.C1C
        pseudorange_l2 = group_sub.C2W
        p1 = pseudorange_l1.values[:,np.newaxis] + sat_bias - b
        p2 = pseudorange_l2.values[:,np.newaxis] + sat_bias - b
        
        ### Carrier Phases Measurements
        carrier_l1 = group_sub.L1C
        carrier_l2 = group_sub.L2W
        cp1 = (carrier_l1.values[:,np.newaxis] * l1) + sat_bias - b
        cp2 = (carrier_l2.values[:,np.newaxis] * l2) + sat_bias - b
        
        ### ionospheric-free Pseudorange and Carrier-phase measurements
        p_if = (((f1**2) / ((f1**2) - (f2**2))) * p1) - (((f2**2) / ((f1**2) - (f2**2))) * p2)
        cp_if = ((((f1**2) / ((f1**2) - (f2**2))) * cp1) - (((f2**2) / ((f1**2) - (f2**2))) * cp2))
        
        ## Wide-lane integer ambiguity
        WL_12 = (cp1/l1 - cp2/l2 - (p1 / l_12)).flatten()[0]
        NW_1 = (((l2/l1)-1)**-1) * ( ((l2/l1)*WL_12) - cp1/l1 + ((l2/l1)*cp2/l2))
        NW_2 = NW_1 - WL_12
        
        ## Narrow-lane integer ambiguity
        NL_12 = (cp1/l1 + cp2/l2 - (p1 / n_12)).flatten()[0]
        NN_1 = (((l2/l1)-1)**-1) * ( ((l2/l1)*NL_12) - cp1/l1 + ((l2/l1)*cp2/l2))
        NN_2 = NN_1 - NL_12
        
        # N = n_12 * (NN_1 + ((l_12/l2) * WL_12))
        N = ( (f1*NL_12)/(f1+f2) ) + ( (f1*f2*WL_12)/(f1**2 - f2**2) )
        cp_if = cp_if - (N * l_if)
        
        # ## ionospheric-free Pseudorange and Carrier-phase measurements
        # N_if = N1_dict[sv_num][0]
        # p_if = p_if + (calcSatBias(rcvr_time,nav,sv)*c) - b
        # cp_if = cp_if + (calcSatBias(rcvr_time,nav,sv)*c) - b
        
        if not isinstance(sp3, pd.DataFrame):
            sat_pos = calcSatPos(rcvr_time, p_if.flatten()[0], nav, sv)
            sat_pos = rot_satpos(sat_pos,p_if)

        else:
            sat_pos = precise_orbit(group_sub.epoch.values[0], p_if, sv, sp3)
            sat_pos = rot_satpos(sat_pos,p_if)
        
        if count == 0:
            H = np.vstack((H,sat_pos.T))
            p_arr[sv_num] = p_if
            cp_arr[sv_num] = cp_if
            sv_arr.append(sv_num)
        
        else:
            azimuth, elevation = calc_az_el(ecef2enu(np.array([x,y,z])[np.newaxis,:],sat_pos.T))
            
            if elevation < 15.:
                continue
            
            else:
                height = cart2ell(np.array([[x,y,z]])).flatten()[-1]
                Tr = saastamoinen(height,elevation[0])
                H = np.vstack((H,sat_pos.T))    
                p_arr[sv_num] = p_if - Tr
                cp_arr[sv_num] = cp_if - Tr
                elev_arr[sv_num] = 0.6 + (0.6/np.sin(np.radians(elevation[0])))
                elev_arr[sv_num+32] = 0.003 + (0.003/np.sin(np.radians(elevation[0])))
                sv_arr.append(sv_num)
    
    if len(H) < 4:
        raise Exception("Less than 4 satellites in view")
    else:
        pass
        
    x_n = init_x
    dt = 0.0 # GPS data frequency 1 data/second
    y_arr = np.vstack([p_arr,cp_arr])
    # elev_arr = np.vstack([elev_arr,elev_arr])
    pos_list = np.where(p_arr != 0)[0].tolist()
    
    if count == 0:
        pass
    
    else:
        R_empt_std = R_std
        
        R_std = elev_arr
        R_std[:32] += R_empt_std ### pseudorange precision
        R_std[32:] += 0.01*l1 ### carrier-phase precision
        R_std = np.where(R_std == 0, np.square(R_empt_std), np.square(R_std))
    
    kf = filter_pos2(x_n, P, y_arr, dt, Q_std, R_std)
    
    for iteration in range(1):
        kf.predict()
        kf.update(z=y_arr,HJacobian=Jacobian_PPP,Hx=Hx_PPP,args=(H,pos_list),hx_args=(H,pos_list))
        
    
    sv_list = np.where(kf.x[8:] != 0)[0].tolist()
    
    for sv in sv_list:
        # if N1_dict[sv][1] - kf.x[8+sv][0] > 5:
        N1_dict[sv][0] = rcvr_time
        N1_dict[sv][1] = kf.x[8+sv][0]/l_if
            
    
    return kf.x, kf.P, kf.log_likelihood, kf.y