# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:14:11 2023

@author: Cholo
"""

import pandas as pd
import numpy as np
from scipy.constants import c
import time

rcvr = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
rcvr.columns = ['time_week','SV','pseudorange','cycle','phase','slipdetect','snr']

eph = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
eph.columns = ['time_week','SV','toc','toe','af0','af1','af2','ura','e','sqrta','dn','m0',
               'w','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','iod']

### coordinates of SV6
sv6 = rcvr[rcvr['SV'] == 6.] 
t =  sv6.time_week.values - sv6.pseudorange.values/c

eph6 = eph[eph['SV'] == 6.]

###coordinate calculation
GM = 3986004.418e8 # Earth's gravitational constant
omegadotE = 7292115.0e-11 # Earth's angular velocity

a = eph6.sqrta.values**2 #semi-major axis
n0 = np.sqrt(GM/a**3) #computed mean motion (rad/sec)

tk = t - eph6.toe.values #time from ephemeris reference epoch

if tk >= 302400:
    tk -= 604800
    
elif tk <= -302400:
    tk += 604800
    
n = n0 + eph6.dn.values #corrected mean motion
Mk = eph6.m0.values + n*tk #Mean anomaly

###calculation of eccentric anomaly
Ek = Mk
f = Mk - (Ek - eph6.e.values*np.sin(Ek))
f_prime = eph6.e.values*np.cos(Ek) - 1 

while abs(f/f_prime) >= 1e-8:
    Ek = Ek - f/f_prime
    f = Mk - (Ek - eph6.e.values*np.sin(Ek))
    f_prime = eph6.e.values*np.cos(Ek) - 1
    
vk = np.arctan2((np.sqrt(1 - np.square(eph6.e.values)) *  np.sin(Ek)), (np.cos(Ek) - eph6.e.values)  ) #true anomaly

phi_k = vk + eph6.w.values #argument of latitude

cus = eph6.cus.values
cuc = eph6.cuc.values

cis = eph6.cis.values
cic = eph6.cic.values

crs = eph6.crs.values
crc = eph6.crc.values

#second harmonic pertubations
delta_uk = cus*np.sin(2*phi_k) + cuc*np.cos(2*phi_k) #argument of latitude correction
delta_rk = crs*np.sin(2*phi_k) + crc*np.cos(2*phi_k) #radius correction
delta_ik = cis*np.sin(2*phi_k) + cic*np.cos(2*phi_k) #inclination correction

uk = phi_k + delta_uk #corrected argument of latitude
rk = a*(1 - eph6.e.values*np.cos(Ek)) + delta_rk #corrected radius
ik = eph6.i0.values + delta_ik + eph6.idot.values*tk #corrected inclination

#position in orbital plane
xk_prime = rk * np.cos(uk)
yk_prime = rk * np.sin(uk)

#corrected longitude of ascending node
omega_k = eph6.omg0.values + (eph6.odot.values - omegadotE)*tk - (omegadotE * eph6.toe.values)

#ECEF coordinates
xk = xk_prime*np.cos(omega_k) - yk_prime*np.cos(ik)*np.sin(omega_k)
yk = xk_prime*np.sin(omega_k) + yk_prime*np.cos(ik)*np.cos(omega_k)
zk = yk_prime*np.sin(ik)