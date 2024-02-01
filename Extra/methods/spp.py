# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:30:59 2024

@author: ASTI
"""

import numpy as np
import numpy.linalg as la
from .sat_pos import calcSatBias, calcSatPos
from .localization import ecef2enu, cart2ell, calc_az_el, rot_satpos, G_mat, est_p, klobuchar
from .EKF import filter_pos, Jacobian, Hx
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
l_12 = c/((f1-f2) * 1e6) ###Wide Lane Combination wavelength
n_12 = c/((f1+f2) * 1e6)

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
                H = np.vstack((H,sat_pos.T))    
                p_arr = np.append(p_arr,p,axis=0)
    
    if len(H) < 4:
        raise Exception("Less than 4 satellites in view")
    else:
        pass
    
    
    x_n = init_x
    dt = 0.0 # GPS data frequency 1 data/second
    
    kf = filter_pos(x_n, P, p_arr, dt, Q_std, R_std)
    b_prior = kf.x_prior.flatten()[3]
    
    for iteration in range(1):
        kf.predict()
        # z = p_arr + b_prior - kf.x.flatten()[3]
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