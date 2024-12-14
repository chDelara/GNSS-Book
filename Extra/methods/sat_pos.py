# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:13:29 2023

@author: Cholo
"""

import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.interpolate import lagrange

def calcSatBias(rcvr_time,eph_df,SV):
    
    t = rcvr_time
    
    eph_n = eph_df[eph_df['SV'] == SV]
    epoch_idx = abs(eph_n.toe - t).idxmin()
    eph_n = eph_n.loc[epoch_idx:epoch_idx,:]
    
    ###coordinate calculation
    GM = 3986004.418e8 # Earth's gravitational constant
    omegadotE = 7292115.1467e-11 # Earth's angular velocity
    
    a = eph_n.sqrta.values**2 #semi-major axis
    e = eph_n.e.values
    n0 = np.sqrt(GM/a**3) #computed mean motion (rad/sec)
    
    tk = t - eph_n.toe.values #time from ephemeris reference epoch
    
    if tk >= 302400:
        tk -= 604800
        
    elif tk <= -302400:
        tk += 604800
        
    n = n0 + eph_n.dn.values #corrected mean motion
    Mk = eph_n.m0.values + n*tk #Mean anomaly
    
    ###calculation of eccentric anomaly
    Ek = Mk
    f = Mk - (Ek - e*np.sin(Ek))
    f_prime = e*np.cos(Ek) - 1 
    
    while abs(f/f_prime) >= 1e-8:
        Ek = Ek - f/f_prime
        f = Mk - (Ek - e*np.sin(Ek))
        f_prime = e*np.cos(Ek) - 1
    
    ###calculation of relativistic corrections
    F = (-2 * np.sqrt(GM)) / c**2
    tr = F * e * np.sqrt(a) * np.sin(Ek)

    ###calculation of SV PRN code phase offset (satellite clock bias)
    toe = eph_n.toe.values
    af0 = eph_n.af0.values
    af1 = eph_n.af1.values
    af2 = eph_n.af2.values

    d_tsv = af0 + (af1*(t - toe)) + (af2*(t - toe)**2) + tr

    return d_tsv

def calcSatPos(rcvr_time,pseudorange,eph_df,SV):
    
    t =  rcvr_time - pseudorange/c
    
    eph_n = eph_df[eph_df['SV'] == SV]
    epoch_idx = abs(eph_n.toe - t).idxmin()
    eph_n = eph_n.loc[epoch_idx:epoch_idx,:]
    
    ###coordinate calculation
    GM = 3986004.418e8 # Earth's gravitational constant
    omegadotE = 7292115.1467e-11 # Earth's angular velocity
    
    a = eph_n.sqrta.values**2 #semi-major axis
    e = eph_n.e.values
    n0 = np.sqrt(GM/a**3) #computed mean motion (rad/sec)
    
    tk = t - eph_n.toe.values #time from ephemeris reference epoch
  
    if tk >= 302400:
        tk -= 604800
        
    elif tk <= -302400:
        tk += 604800
    
    n = n0 + eph_n.dn.values #corrected mean motion
    Mk = eph_n.m0.values + n*tk #Mean anomaly
    
    ###calculation of eccentric anomaly
    Ek = Mk
    f = Mk - (Ek - e*np.sin(Ek))
    f_prime = e*np.cos(Ek) - 1 
    
    while abs(f/f_prime) >= 1e-8:
        Ek = Ek - f/f_prime
        f = Mk - (Ek - e*np.sin(Ek))
        f_prime = e*np.cos(Ek) - 1
    
    vk = np.arctan2((np.sqrt(1 - np.square(e)) *  np.sin(Ek)), (np.cos(Ek) - e)  ) #true anomaly

    phi_k = vk + eph_n.omega.values #argument of latitude

    cus = eph_n.cus.values
    cuc = eph_n.cuc.values

    cis = eph_n.cis.values
    cic = eph_n.cic.values

    crs = eph_n.crs.values
    crc = eph_n.crc.values

    #second harmonic pertubations
    delta_uk = cus*np.sin(2*phi_k) + cuc*np.cos(2*phi_k) #argument of latitude correction
    delta_rk = crs*np.sin(2*phi_k) + crc*np.cos(2*phi_k) #radius correction
    delta_ik = cis*np.sin(2*phi_k) + cic*np.cos(2*phi_k) #inclination correction

    uk = phi_k + delta_uk #corrected argument of latitude
    rk = a*(1 - eph_n.e.values*np.cos(Ek)) + delta_rk #corrected radius
    ik = eph_n.i0.values + delta_ik + eph_n.idot.values*tk #corrected inclination

    #position in orbital plane
    xk_prime = rk * np.cos(uk)
    yk_prime = rk * np.sin(uk)

    #corrected longitude of ascending node
    omega_k = eph_n.omg0.values + (eph_n.odot.values - omegadotE)*tk - (omegadotE * eph_n.toe.values)

    #ECEF coordinates
    xk = xk_prime*np.cos(omega_k) - yk_prime*np.cos(ik)*np.sin(omega_k)
    yk = xk_prime*np.sin(omega_k) + yk_prime*np.cos(ik)*np.cos(omega_k)
    zk = yk_prime*np.sin(ik)
    
    return np.array([xk,yk,zk])

def precise_orbit(epoch, pseudorange, sv, sp3):
    
    ### set-up x,y,z, and epoch values for interpolation
    sp3_sub = sp3[sp3.SV == sv]
    
    t = (sp3_sub.epoch.values.astype(np.datetime64) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
    ep = pd.Timestamp(epoch).timestamp() - (pseudorange/c) - t[0]
    t -= t[0]
    
    idx = np.argmin(abs(t - ep))
    
    x = sp3_sub.x.values.astype(float)
    y = sp3_sub.y.values.astype(float)
    z = sp3_sub.z.values.astype(float)
    
    ###Langrange Interpolation
    polyx = lagrange(t[idx-2:idx+2],x[idx-2:idx+2])
    polyy = lagrange(t[idx-2:idx+2],y[idx-2:idx+2])
    polyz = lagrange(t[idx-2:idx+2],z[idx-2:idx+2])
    
    x_interp = polyx(ep) * 1000
    y_interp = polyy(ep) * 1000
    z_interp = polyz(ep) * 1000
    
    
    # x_interp = np.interp(pd.Timestamp(epoch).timestamp() - pseudorange/c,t,x.astype(float)) * 1000
    # y_interp = np.interp(pd.Timestamp(epoch).timestamp() - pseudorange/c,t,y.astype(float)) * 1000
    # z_interp = np.interp(pd.Timestamp(epoch).timestamp() - pseudorange/c,t,z.astype(float)) * 1000
    return np.vstack([x_interp,
                      y_interp,
                      z_interp])