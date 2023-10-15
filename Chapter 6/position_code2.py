# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:21:01 2023

@author: Cholo
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

from scipy.constants import c
from methods.ephemeris_calc import *
from methods.pos_calc import *


###input for initial position
rwy30Start = np.array([[-2694685.473,-4293642.366,3857878.924]])
ref = np.array([[-2700400,-4292560.,3855270.]])
ref_ell = cart2ell(ref).T

rcvr = pd.read_csv(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Stanford/September_18_2000/r091800a.dat',header=None,
                   delim_whitespace=True).dropna()
# rcvr = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Stanford/September_18_2000/r091800a.dat',header=None,
#                    delim_whitespace=True).dropna()
rcvr.columns = ['time_week','SV','pseudorange','cycle','phase','slipdetect','snr']

eph = pd.read_csv(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Stanford/September_18_2000/e091800a.dat',header=None,
                  delim_whitespace=True).dropna()
# eph = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Original/eph.dat',header=None).dropna()
eph.columns = ['time_week','SV','toc','toe','af0','af1','af2','ura','e','sqrta','dn','m0',
                'w','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','iod']


# init_x = np.append(rwy30Start,0)
init_x = np.array([0,0,0,0])
sample = rcvr.groupby("time_week")

time_list, e_list, n_list, u_list, b_list = [], [], [], [], []

start = time.time()
for rcvr_time, group in sample:
    
    print(f"Epoch: {rcvr_time}")
    H = np.empty((0,3))
    p_arr = np.empty((0,1))
    
    for sv in group.SV.unique():
        sat_pos = calcSatPos(group, eph, rcvr_time, sv)
        p = group[group.SV == sv].pseudorange.values[:,np.newaxis] + (calcSatBias(group,eph,rcvr_time,sv) - init_x[-1]/c)*c
        
        # sat_pos = rot_satpos(sat_pos,p)
        H = np.vstack((H,sat_pos.T))
        
        p_arr = np.append(p_arr,p,axis=0)
    
    x = init_x
    delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (p_arr - est_p(x,H))
    x = init_x + delta_xb.flatten()
    
    counter = 0
    while ((np.abs(delta_xb) > 1e-6).all()) & (counter <= 100):
        delta_xb = la.inv(G_mat(x,H).T @ G_mat(x,H)) @ G_mat(x,H).T @ (p_arr - est_p(x,H) + init_x[-1] - x[-1])
        x = x + delta_xb.flatten()
        counter += 1
    
    lat, lon, height = cart2ell(x[:-1]).flatten()    
    print(f"Latitude: {lat}, Longitude: {lon}, Ellipsoidal Height: {height}")
    print(f"Receiver Clock Bias: {x[-1]/c} s\n")
    
    e, n, u = ecef2enu(ref,x[:3][np.newaxis,:]).flatten()
    b = x[-1]/c
    
    time_list.append(rcvr_time)
    e_list.append(e)
    n_list.append(n)
    u_list.append(u)
    b_list.append(b)
    
    # init_x = x

end = time.time()
print(f"Runtime: {end-start} seconds")

condition = np.where((np.abs(np.array(e_list)) < 1000) & (np.abs(np.array(n_list)) < 1000) & (np.abs(np.array(u_list)) < 1000))
time_list2 = np.array(time_list)[condition]
e_list2 = np.array(e_list)[condition]
n_list2 = np.array(n_list)[condition]
u_list2 = np.array(u_list)[condition]
b_list2 = np.array(b_list)[condition]

###plot satellite position during GNSS measurement
fig,ax = plt.subplots(figsize=(12,7),dpi=120)
sc = ax.scatter(x=e_list2,y=n_list2,
                c=time_list2,
                cmap='rainbow_r')

fig.colorbar(sc)
ax.set_xlim(np.min(e_list2),np.max(e_list2))
ax.set_ylim(np.min(n_list2),np.max(n_list2))
ax.set_title('GNSS Measurement')
plt.show()

###plot each component versus time 
fig,ax = plt.subplots(nrows=4,figsize=(12,14),dpi=120)
ax[0].scatter(x=time_list2,y=e_list2,c = 'black',label = 'Easting')
ax[0].set_title(f'Easting vs time, Mean: {e_list2.mean()} Standard Deviation: {e_list2.std()}')
ax[0].legend()


ax[1].scatter(x=time_list2,y=n_list2,c = 'red',label = 'Northing')
ax[1].set_title(f'Northing vs time, Mean: {n_list2.mean()} Standard Deviation: {n_list2.std()}')
ax[1].legend()


ax[2].scatter(x=time_list2,y=u_list2,c = 'blue',label = 'Up')
ax[2].set_title(f'Up vs time, Mean: {u_list2.mean()} Standard Deviation: {u_list2.std()}')
ax[2].legend()


ax[3].scatter(x=time_list2,y=b_list2,c = 'green',label = 'User Clock Bias')
ax[3].set_title(f'Receiver Clock Bias vs time, Mean: {b_list2.mean()} Standard Deviation: {b_list2.std()}')
ax[3].legend()
plt.show()