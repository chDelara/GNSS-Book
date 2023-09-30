# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:12:28 2023

@author: ASTI
"""

import pandas as pd
import numpy as np
from scipy.constants import c
import time

rcvr = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
# rcvr = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/rcvr.dat',header=None).dropna()
rcvr.columns = ['time_week','SV','pseudorange','cycle','phase','slipdetect','snr']

eph = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
# eph = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
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
e = eph6.e.values
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
toc = eph6.toc.values
af0 = eph6.af0.values
af1 = eph6.af1.values
af2 = eph6.af2.values

d_tsv = af0 + af1*(t - toc) + (af2*(t-toc)**2) + tr

corr_t = t - d_tsv