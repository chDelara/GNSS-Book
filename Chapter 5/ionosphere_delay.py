# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:06:37 2023

@author: Cholo
"""

import pandas as pd
import matplotlib.pyplot as plt
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

###for reading the ionospheric corrections only
iono_corr = gr.load(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/September_18_2000/pp091800.nav',use='G').ionospheric_corr_GPS
# iono_corr = gr.load(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/pp091800.nav',use='G')

# def klobuchar(iono_corr, user_loc, az, el, rcvr_time):

###user defined variable
sv = 5
ref = np.array([[-2725252.3706,-4295978.5064,3833958.9487]])
ref_ell = cart2ell(ref).T

rio = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rio',header=None)
# rio = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rio',header=None)
rio.columns = ["GPS_week","rcvr_tow","SV","C1","L1","L2","P1","P2","D1","D2"]

rin = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rin',header=None)
# rin = pd.read_fwf(r'C:/Users/ASTI/Desktop/GNSS/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rin',header=None)
rin.columns = ["SV","m0","dn","e","sqrta","omg0","i0","w","odot","idot","cuc","cus","crc","crs","cic","cis","toe","iode",
               "GPS_week","toc","af0","af1","af2","wdot"]

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

lat_list,lon_list, az_list, el_list = [], [], [], []
for date in rio_svn['GPS_date']:
    print("GPS Time: ",date)
    sat_pos = calcSatPos(rio,rin,date,"P1",sv).T
    lat, lon, height = cart2ell(sat_pos).flatten()
    lat_list.append(lat)
    lon_list.append(lon)
    print("{lat},{lon}")
    
    ###compute azimuth and elevation angle from 
    az,el = calc_az_el(ecef2enu(ref,sat_pos)).flatten()
    az_list.append(az)
    el_list.append(el)
    
    print(f'Azimuth: {az}')
    print(f'Elevation Angle: {el}')
    
    
rio_svn['zenith'] = el_list
rio_svn['zenith'] = 90 - rio_svn['zenith']

OF = np.sqrt(1 - np.square(a*np.sin(np.radians(rio_svn['zenith'].values)) / (a + 350000)) )
rio_svn['OF'] = OF
rio_svn['zen_delay'] = rio_svn['io_delay'] / rio_svn['OF']


date_list = rio[rio.SV == sv]['GPS_date'].tolist()
###plot satellite position during GNSS measurement
fig,ax = plt.subplots(figsize=(12,7),dpi=120)
plt.scatter(x=lon_list,y=lat_list)

###ref point coordinates
x,y = ref_ell.flatten()[:2]
plt.scatter(x=y,y=x,c='red')
plt.title('Satellite Footprint')
plt.show()

###plot elevation and azimuth angle over time
fig1,ax1 = plt.subplots(nrows=2,figsize=(12,7),dpi=120,sharex=True)
ax1[0].scatter(x=date_list,y=az_list)
ax1[0].set_title('Satellite Azimuth vs Time')

ax1[1].scatter(x=date_list,y=el_list,c="red")
ax1[1].set_title('Satellite Elevation vs Time')
plt.show()

###plot zenith delay over time
fig,ax2 = plt.subplots(figsize=(12,7),dpi=120)
ax2.scatter(x=date_list,y=rio_svn.zen_delay)
ax2.set_title('Zenith Ionospheric Delay Overtime')
plt.show()