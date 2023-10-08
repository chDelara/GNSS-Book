# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:27:19 2023

@author: ASTI
"""

import pandas as pd
import numpy as np
from scipy.constants import c
import time

# ###input for transformation functions
# rwy30Start = np.array([[-2694685.473,-4293642.366,3857878.924]])
# rwy30End = np.array([[-2694892.460,-4293083.225,3858353.437]])

# rcvr = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
# # rcvr = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
# rcvr.columns = ['time_week','SV','pseudorange','cycle','phase','slipdetect','snr']

# eph = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
# # eph = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
# eph.columns = ['time_week','SV','toc','toe','af0','af1','af2','ura','e','sqrta','dn','m0',
#                'w','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','iod']

def ell2cart(point):
    """
    

    Parameters
    ----------
    point : numpy.array
        an array containing coordinates of a point in WGS84 coordinates.

    Returns
    -------
    cart_point : numpy.array
        an array containing transformed coordinates from WGS84 to ECEF coordinates

    """
    
    lat, lon, h = point.flatten()
    
    # WGS84 parameters
    a = 6378137.0 # semi-major axis
    f = 1/298.257223563 # inverse flattening
    b = a - (a*f) # semi-minor axis
    e = np.sqrt(2*f - f**2)
    omega_E = 7292115.0e-11 # Earth's angular velocity
    GM = 3986004.418e8 # Earth's gravitational constant
    
    
    N = a / np.sqrt(1 - (e**2) * np.sin(np.radians(lat))**2)
    
    cart_point = [ [(N + h) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))],
                  [(N + h) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))],
                  [((N*(1-e**2)) + h) * np.sin(np.radians(lat))] ]
    
    return np.array(cart_point)

def cart2ell(point):
    # WGS84 parameters
    a = 6378137.0 # semi-major axis
    f = 1/298.257223563 # inverse flattening
    b = a - (a*f) # semi-minor axis
    e = np.sqrt(2*f - f**2)
    omega_E = 7292115.0e-11 # Earth's angular velocity
    GM = 3986004.418e8 # Earth's gravitational constant
    
    x, y, z = point.flatten()
    lon = np.degrees(np.arctan2(y,x))
    
    p = np.sqrt(x**2 + y**2)
    lat0 = np.arctan(z/((1-e**2)*p))
    N = a / (np.sqrt(1 - ((e**2) * np.sin(lat0)**2)))
    h0 = (p/np.cos(lat0)) - N
    
    delta_lat = 1000.0
    delta_h = 1000.0

    while (delta_lat >= 1e-7) | (delta_h >= 1e-4):
        N_d = np.sqrt(1 - ((e**2) * np.sin(lat0)**2))
        N = a / N_d
        
        h1 = (p/np.cos(lat0)) - N
        lat1 = np.arctan( z/ (p * (1 - (e**2 * (N/(N+h1))))) )
        
        delta_lat = abs(lat1 - lat0)
        delta_h = abs(h1 - h0)
        
        lat0, h0 = lat1, h1

    return np.array([[np.degrees(lat0),lon,h0]]).T

def ecef2enu(point_origin,point):
    lat, lon, h = cart2ell(point_origin).flatten()
    rot_mat = np.array([[-np.sin(np.radians(lon)),np.cos(np.radians(lon)),0],
                        [-np.sin(np.radians(lat))*np.cos(np.radians(lon)),-np.sin(np.radians(lat))*np.sin(np.radians(lon)),np.cos(np.radians(lat))],
                        [np.cos(np.radians(lat))*np.cos(np.radians(lon)),np.cos(np.radians(lat))*np.sin(np.radians(lon)),np.sin(np.radians(lat))]])
    
    return (rot_mat @ point.T) - (rot_mat @ point_origin.T)

def calc_az_el(point):
    e,n,u = point.flatten()
    dist = np.sqrt(np.square(point).sum())
    
    az = np.degrees(np.arctan2(e,n))
    el = np.degrees(np.arcsin(u/dist))
    
    return np.array([[az],
                    [el]])

def calcSatBias(rcvr_df,eph_df,freq,SV):

    ### coordinates of SV
    sv_n = rcvr_df[rcvr_df['SV'] == SV]
    
    try:
        t =  sv_n.time_week.values - sv_n.pseudorange.values/c
    except:
        t = sv_n.rcvr_tow.values - sv_n[freq].values/c
    
    eph_n = eph_df[eph_df['SV'] == SV]
    
    ###coordinate calculation
    GM = 3986004.418e8 # Earth's gravitational constant
    omegadotE = 7292115.0e-11 # Earth's angular velocity
    
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
    toc = eph_n.toc.values
    af0 = eph_n.af0.values
    af1 = eph_n.af1.values
    af2 = eph_n.af2.values

    d_tsv = af0 + af1*(t - toc) + (af2*(t-toc)**2) + tr

    return d_tsv

def calcSatPos(rcvr_df,eph_df,freq,SV):

    ### coordinates of SV
    sv_n = rcvr_df[rcvr_df['SV'] == SV]
    
    try:
        t =  sv_n.time_week.values - sv_n.pseudorange.values/c - calcSatBias(rcvr_df,eph_df,freq,SV)
    except:
        t = sv_n.rcvr_tow.values - sv_n[freq].values/c - calcSatBias(rcvr_df,eph_df,freq,SV)
    
    eph_n = eph_df[eph_df['SV'] == SV]
    
    ###coordinate calculation
    GM = 3986004.418e8 # Earth's gravitational constant
    omegadotE = 7292115.0e-11 # Earth's angular velocity
    
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

    phi_k = vk + eph_n.w.values #argument of latitude

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

def calcSatVel(rcvr_df,eph_df,SV):
    ### coordinates of SV
    sv_n = rcvr_df[rcvr_df['SV'] == SV] 
    t =  sv_n.time_week.values - sv_n.pseudorange.values/c - calcSatBias(rcvr_df,eph_df,SV)
    
    eph_n = eph_df[eph_df['SV'] == SV]
    
    ###coordinate calculation
    GM = 3986004.418e8 # Earth's gravitational constant
    omegadotE = 7292115.0e-11 # Earth's angular velocity
    
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
    
    vx_inertial = n*a / (1 - e*np.cos(Ek)) * -np.sin(Ek)
    vy_inertial = n*a / (1 - e*np.cos(Ek)) * np.sqrt(1 - e**2) * np.cos(Ek)
        
    vk = np.arctan2((np.sqrt(1 - np.square(eph_n.e.values)) *  np.sin(Ek)), (np.cos(Ek) - eph_n.e.values)  ) #true anomaly
    
    phi_k = vk + eph_n.w.values #argument of latitude
    
    #second harmonic pertubations
    cis = eph_n.cis.values
    cic = eph_n.cic.values
    
    delta_ik = cis*np.sin(2*phi_k) + cic*np.cos(2*phi_k) #inclination correction
    
    
    #calculation of correct inclination
    ik = eph_n.i0.values + delta_ik + eph_n.idot.values*tk #corrected inclination
    
    #corrected longitude of ascending node
    omega_k = eph_n.omg0.values + (eph_n.odot.values - omegadotE)*tk - (omegadotE * eph_n.toe.values)
    
    #ECEF coordinates
    xk = vx_inertial*np.cos(omega_k) - vy_inertial*np.cos(ik)*np.sin(omega_k)
    yk = vx_inertial*np.sin(omega_k) + vy_inertial*np.cos(ik)*np.cos(omega_k)
    zk = vy_inertial*np.sin(ik)
     
    return np.array([xk,yk,zk])