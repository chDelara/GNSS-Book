# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:55:13 2023

@author: Cholo
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.constants import c

import sys
sys.path.insert(0,r'D:/Cholo/Project/GNSS/Per Enge Book Codes/GNSS-Book/Chapter 4')
# sys.path.insert(0,r'C:\Users\ASTI\Desktop\GNSS\codes\GPS Book Homework\GNSS-Book\Chapter 4')
from all_func import ecef2enu, calc_az_el, cart2ell, ell2cart

###input for transformation functions
rwy30Start = np.array([[-2694685.473,-4293642.366,3857878.924]])
rwy30End = np.array([[-2694892.460,-4293083.225,3858353.437]])

rcvr = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
# rcvr = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
rcvr.columns = ['time_week','SV','pseudorange','cycle','phase','slipdetect','snr']

eph = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
# eph = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
eph.columns = ['time_week','SV','toc','toe','af0','af1','af2','ura','e','sqrta','dn','m0',
                'w','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','iod']


def calcSatBias(rcvr_df,eph_df,SV):
    
    ### coordinates of SV
    sv_n = rcvr_df[rcvr_df['SV'] == SV] 
    t =  sv_n.time_week.values - sv_n.pseudorange.values/c
    
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

def calcSatPos(rcvr_df,eph_df,SV):
    
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

def G_mat(init_x, G):
    x, y, z, b = init_x
    
    d_arr = np.sqrt(np.sum(np.square(G - [x,y,z]),axis=1)[:,np.newaxis]) + b

    G = -(G-[x,y,z])/d_arr
    G = np.append(G,np.ones(shape=(len(G),1)),axis=1)
    
    return G
    

def est_p(init_x,G):
    return np.sqrt(np.sum(np.square(G - init_x[:-1]),axis=1)[:,np.newaxis]) + init_x[-1]

def rot_satpos(sat_pos,pseudorange):
    omegadotE = 7292115.0e-11 # Earth's angular velocity
    
    pseudorange = pseudorange.flatten()[0]
    rot_mat = np.array([[np.cos(omegadotE * pseudorange/c), np.sin(omegadotE * pseudorange/c), 0],
                        [-np.sin(omegadotE * pseudorange/c), np.cos(omegadotE * pseudorange/c), 0],
                        [0, 0, 1]])
    
    return rot_mat @ sat_pos

init_x = np.append(rwy30Start,0)

H = np.empty((0,3))
p_arr = np.empty((0,1))
for sv in rcvr.SV.unique():
    sat_pos = calcSatPos(rcvr, eph, sv)
    p = rcvr[rcvr.SV == sv].pseudorange.values[:,np.newaxis] + calcSatBias(rcvr,eph,sv)*c
    
    # sat_pos = rot_satpos(sat_pos,p)
    H = np.vstack((H,sat_pos.T))
    
    p_arr = np.append(p_arr,p,axis=0)
    
delta_xb = la.inv(G_mat(init_x,H).T @ G_mat(init_x,H)) @ G_mat(init_x,H).T @ (p_arr - est_p(init_x,H))

while ~(np.abs(delta_xb) < 1e-6).all():
    init_x = init_x + delta_xb.flatten()
    delta_xb = la.inv(G_mat(init_x,H).T @ G_mat(init_x,H)) @ G_mat(init_x,H).T @ (p_arr - est_p(init_x,H))
    print(delta_xb)
    print(init_x,"\n")