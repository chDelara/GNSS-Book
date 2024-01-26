# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:06:44 2023

@author: ASTI
"""

import pandas as pd
import numpy as np


"""
H = np.empty((0,3))
p_arr = np.empty((0,1))
group = group.dropna(subset=['C1C','C2W'])

for sv in group.SV.unique():
    group_sub = group[group.SV == sv]
    # pseudorange = (((f1**2) / ((f1**2) - (f2**2))) * group_sub.C1C) - (((f2**2) / ((f1**2) - (f2**2))) * group_sub.C2W)
    pseudorange = group_sub.C1C
    pseudorange2 = group_sub.C2W
    p = pseudorange.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
    p2 = pseudorange2.values[:,np.newaxis] + (calcSatBias(rcvr_time,nav,sv) - init_x[-1]/c)*c
    
    sat_pos = calcSatPos(rcvr_time, p.flatten()[0], nav, sv)
    sat_pos = rot_satpos(sat_pos,p)
    
    if count == 0:
        H = np.vstack((H,sat_pos.T))    
        p_arr = np.append(p_arr,p,axis=0)
        p_arr  = np.append(p_arr,p2,axis=0)
    
    else:
        azimuth, elevation = calc_az_el(ecef2enu(np.array(init_x[:-1])[np.newaxis,:],sat_pos.T))
        
        if elevation < 5.:
            continue
        else:
            H = np.vstack((H,sat_pos.T))    
            p_arr = np.append(p_arr,p,axis=0)
            p_arr  = np.append(p_arr,p2,axis=0)

if len(H) < 4:
    raise Exception("Less than 4 satellites in view")
else:
    pass
"""

def klobuchar(iono_corr, user_loc, az, el, rcvr_time):
    ### https://gssc.esa.int/navipedia/index.php/Klobuchar_Ionospheric_Model
    
    # satellite transmitted ionospheric correction coefficients
    alpha_n = iono_corr[:4]
    beta_n = iono_corr[4:]
    
    lat, lon, height = user_loc.flatten()
    az = np.radians(az)
    el = np.radians(el)
    
    # Calculation of the earth-centered angle
    phi = (0.0137 / (el + 0.11)) - 0.022
    
    # Computation of the latitude of the ionospheric piercing point (IPP)
    lat_I = lat + phi * np.cos(az)
    
    if lat_I > 0.416:
        lat_I = 0.416
    elif lat_I < -0.416:
        lat_I = -0.416
    else:
        pass
    
    # Compute the longitude of the ionospheric piercing point (IPP)
    lon_I = lon + (phi * np.sin(az)/np.cos(lat_I))
    
    # find the geomagnetic latitude of the ionospheric piercing point (IPP)
    phi_m = lat_I + (0.064*np.cos(lon_I - 1.617))
    
    # find the local time at IPP
    t = (43200 * lon_I) + rcvr_time
    
    if t >= 86400:
        t -= 86400
    elif t < 0:
        t += 86400
    else:
        pass
    
    # computation of the amplitude of ionospheric delay
    A_I = 0
    
    for counter in range(4):
        A_I += alpha_n[counter] * phi_m**counter
    
    if A_I < 0:
        A_I = 0
    else:
        pass
    
    # computation of the period of ionospheric delay
    P_I = 0
    
    for counter in range(4):
        P_I += beta_n[counter] * phi_m**counter
        
    if P_I < 72000:
        P_I = 72000
    else:
        pass
    
    # computation of the phase of ionospheric delay
    X_I = 2*np.pi * (t - 50400) / P_I
    
    # computation of the slant factor
    F = 1.0 + (16.0 * (0.53 - el)**3)
    
    I_Ln = 0
    
    if abs(X_I) <= 1.57:
        I_Ln = (5e-9 + (A_I * (1 - ((X_I**2)/2) + ((X_I**4)/24)))) * F
    else:
        I_Ln = 5e-9 * F
    
    return I_Ln
    
    