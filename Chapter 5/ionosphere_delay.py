# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:06:37 2023

@author: Cholo
"""

import pandas as pd
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import sys

import georinex as gr

sys.path.insert(0,r'D:/Cholo/Project/GNSS/Per Enge Book Codes/GNSS-Book/Chapter 4')
# sys.path.insert(0,r'C:\Users\ASTI\Desktop\GNSS\codes\GPS Book Homework\GNSS-Book\Chapter 4')
from all_func import *

gps_date_start = datetime(1980,1,6)
f1 = 1575.42 #MHz
f2 = 1227.6 #MHz
f5 = 1176. #MHz
a = 6378137.0 # semi-major axis

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

###for reading the ionospheric corrections only
iono_corr = gr.load(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/July_15_2000/pp071500.nav',use='G').ionospheric_corr_GPS
# iono_corr = gr.load(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/pp091800.nav',use='G')

    
## September 18, 2000 data
##user defined variable
sv = 3
ref = np.array([[-2725252.3706,-4295978.5064,3833958.9487]])
ref_ell = cart2ell(ref).T

rio = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rio',header=None)
# rio = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rio',header=None)
rio.columns = ["GPS_week","rcvr_tow","SV","C1","L1","L2","P1","P2","D1","D2"]

rin = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rin',header=None)
# rin = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rin',header=None)
rin.columns = ["SV","m0","dn","e","sqrta","omg0","i0","w","odot","idot","cuc","cus","crc","crs","cic","cis","toe","iode",
                "GPS_week","toc","af0","af1","af2","wdot"]

# ### July 15, 2000 data
# ###user defined variable
# sv = 2
# ref = np.array([[-2725252.3748, -4295978.5284, 3833958.9631]])
# ref_ell = cart2ell(ref).T

# rio = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/July_15_2000/Parsed/pp071500.rio',header=None)
# # rio = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rio',header=None)
# rio.columns = ["GPS_week","rcvr_tow","SV","C1","L1","L2","P1","P2","D1","D2"]

# rin = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/July_15_2000/Parsed/pp071500.rin',header=None)
# # rin = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rin',header=None)
# rin.columns = ["SV","m0","dn","e","sqrta","omg0","i0","w","odot","idot","cuc","cus","crc","crs","cic","cis","toe","iode",
#                 "GPS_week","toc","af0","af1","af2","wdot"]

###calculating GPS date for each measurement 
gps_date_list = []
for idx, row in rio.iterrows():
    week = row.GPS_week
    sec = row.rcvr_tow
    gps_date = gps_date_start + timedelta(weeks=week) + timedelta(seconds=sec)
    gps_date_list.append(gps_date)
end = time.time()

rio.insert(loc=0,column="GPS_date",value=gps_date_list)

gps_date_list = []
for idx, row in rin.iterrows():
    week = row.GPS_week
    sec = row.toe
    gps_date = gps_date_start + timedelta(weeks=week) + timedelta(seconds=sec)
    gps_date_list.append(gps_date)
end = time.time()

rin.insert(loc=0,column="GPS_date",value=gps_date_list)

io_delay = ((f2**2) / ((f1**2) + (f2**2))) * (rio.P2 - rio.P1)
rio['io_delay'] = io_delay

rio_svn = rio[rio.SV == sv].reset_index(drop=True)

lat_list,lon_list, az_list, el_list, klob_list = [], [], [], [], []
start_rcvr_time = rio_svn['rcvr_tow'].iloc[0]
for date in rio_svn['GPS_date']:
    print("GPS Time: ",date)
    sat_pos = calcSatPos(rio,rin,date,"P1",sv).T
    lat, lon, height = cart2ell(sat_pos).flatten()
    lat_list.append(lat)
    lon_list.append(lon)
    print(f"{lat},{lon}")
    
    ###compute azimuth and elevation angle from 
    az,el = calc_az_el(ecef2enu(ref,sat_pos)).flatten()
    az_list.append(az)
    el_list.append(el)
    
    print(f'Azimuth: {az}')
    print(f'Elevation Angle: {el}')
    
    ###compute klobuchar model
    rcvr_time = rio_svn[rio_svn['GPS_date'] == date].rcvr_tow.values[0]
    
    # if rcvr_time < start_rcvr_time:
    #     rcvr_time += start_rcvr_time
        
    #     start_rcvr_time = rcvr_time
    # else:
    #     start_rcvr_time = rcvr_time
    
    I = klobuchar(iono_corr, ref_ell, az, el, rcvr_time) * c
    klob_list.append(I)
    

###Appending ionospheric zenith delay to observation dataframe
rio_svn['zenith'] = el_list
rio_svn['zenith'] = 90 - rio_svn['zenith']

OF = 1/np.sqrt(1 - np.square(a*np.sin(np.radians(rio_svn['zenith'].values)) / (a + 350000)) )
rio_svn['OF'] = OF
rio_svn['zen_delay'] = rio_svn['io_delay'] / rio_svn['OF']

###Appending klobuchar ionospheric model to observation dataframe
rio_svn['klobuchar'] = klob_list
rio_svn['zen_delay2'] = rio_svn['klobuchar'] / rio_svn['OF']

date_list = rio[rio.SV == sv]['GPS_date'].tolist()
###plot satellite position during GNSS measurement
fig,ax = plt.subplots(figsize=(12,7),dpi=120)
sc = ax.scatter(x=lon_list,y=lat_list,
                c=[mdates.date2num(date) for date in date_list],
                cmap='viridis')

loc = mdates.AutoDateLocator()
fig.colorbar(sc,ticks=loc,format=mdates.AutoDateFormatter(loc))

###ref point coordinates
x,y = ref_ell.flatten()[:2]
ax.scatter(x=y,y=x,c='red')
ax.set_title('Satellite Footprint')
plt.show()

###plot elevation and azimuth angle over time
fig1,ax1 = plt.subplots(nrows=2,figsize=(12,7),dpi=120,sharex=True)
ax1[0].scatter(x=date_list,y=az_list)
ax1[0].set_title('Satellite Azimuth vs Time')

ax1[1].scatter(x=date_list,y=el_list,c="red")
ax1[1].set_title('Satellite Elevation vs Time')
plt.show()

###plot L1-L2 and klobuchar zenith delay over time
fig,ax2 = plt.subplots(figsize=(12,7),dpi=120)
ax2.scatter(x=date_list,y=rio_svn.zen_delay,c='red',label='L1-L2 measurements')
ax2.scatter(x=date_list,y=rio_svn.zen_delay2,c='black',label='Klobuchar model')
plt.legend()
ax2.set_title('L1-L2 vs Klobuchar Ionospheric Delay Overtime')
plt.show()