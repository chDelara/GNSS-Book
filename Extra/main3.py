# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:18:42 2024

@author: ASTI
"""

import georinex as gr
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.constants import c
from methods.localization import ell2cart, cart2ell, ecef2enu
from methods.spp import calc_rcvr_pos3, calc_rcvr_pos4, calc_cp_pos, calc_cp_pos2, calc_cp_pos3
from datetime import datetime,timedelta
import time

gps_date_start = datetime(1980,1,6)
f1 = 1575.42 #MHz
f2 = 1227.6 #MHz
f5 = 1176. #MHz

l1 = c / (f1 * 1e6) ###L1 wavelength
l2 = c / (f2 * 1e6) ###L2 wavelength
l5 = c / (f5 * 1e6) ###L5 wavelength
l_12 = c/((f1-f2) * 1e6) ###Wide Lane Combination wavelength
n_12 = c/((f1+f2) * 1e6)

###Molave
# Laptop
# obs = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/Molave/IGS000USA_R_20193020215_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/Molave/IGS000USA_R_20193020215_00M_01S_MN.rnx',use="G")
# p_orb = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/sp3/GFZ0MGXRAP_20193020000_01D_05M_ORB.SP3/GFZ0MGXRAP_20193020000_01D_05M_ORB.SP3').to_dataframe().reset_index(drop=False)
# p_orb = p_orb[p_orb.sv.str.contains('G',regex=True)]

# PC
# obs = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 1/Molave/Molave/IGS000USA_R_20193020215_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 1/Molave/Molave/IGS000USA_R_20193020215_00M_01S_MN.rnx',use="G")

###Freshie Walk
# obs = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/Freshie/IGS000USA_R_20193010216_24H_15S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/Freshie/IGS000USA_R_20193010216_24H_15S_MN.rnx',use="G")

# PC
# obs = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 4/Freshie_Walk_Day4/IGS000USA_R_20193240054_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 4/Freshie_Walk_Day4/IGS000USA_R_20193240054_00M_01S_MN.rnx',use="G")

###CMC Hill
# Laptop
obs = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/CMC_Hill/IGS000USA_R_20193250055_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
nav = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/CMC_Hill/IGS000USA_R_20193250055_00M_01S_MN.rnx',use="G")

# PC
# obs = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 4/CMC_Hill_Day4/IGS000USA_R_20193250055_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 4/CMC_Hill_Day4/IGS000USA_R_20193250055_00M_01S_MN.rnx',use="G")

### ASTI
# obs = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Data/U-Blox/experiment/sagap2024/gps_20240424.obs',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Data/U-Blox/experiment/sagap2024/gps_20240424.nav',use="G")

### ASTI Rooftop
# obs = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/ASTI Rooftop/gps_20231128 16_41_15-20231128 17_10_19.obs',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/ASTI Rooftop/gps_20231128 16_41_15-20231128 17_10_19.nav',use="G")

### ASTI Flag
# obs = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/ASTI Rooftop/gps_20231128 17_23_20-20231128 17_50_20.obs',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/ASTI Rooftop/gps_20231128 17_23_20-20231128 17_50_20.nav',use="G")

### Davao 1
# obs = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/Davao/gps_20231207 14_18_57-20231207 14_42_23.obs',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/Davao/gps_20231207 14_18_57-20231207 14_42_23.nav',use="G")

### Davao 2
# obs = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/Davao/gps_20231207 16_51_12-20231207 17_05_45.obs',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/Davao/gps_20231207 16_51_12-20231207 17_05_45.nav',use="G")

### Davao 3
# obs = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/Davao/gps_20231210 08_57_08-20231210 09_16_37.obs',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/SAGAP/Codes/GNSS/U-Center/sagap2023/gps/Davao/gps_20231210 08_57_08-20231210 09_16_37.nav',use="G")

###PTAG IGS
# obs = gr.load(r'C:/Users/ASTI/Desktop/GNSS/PTAG00PHL_R_20230180100_01H_30S_MO.crx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/GNSS/PTAG00PHL_R_20230180000_01H_GN.rnx',use="G")

try:
    iono_corr = nav.ionospheric_corr_GPS
except:
    pass

nav = nav.to_dataframe().reset_index(drop=False)


### extract GPS time of the week
obs_columns = obs.columns.tolist()
# obs_columns[:2] = ['SV','epoch']
obs_columns[:2] = ['epoch','SV']
obs.columns = obs_columns

days_epoch = ((np.around((obs.epoch - timedelta(hours=8)).values.astype(np.int64),-9) // 10 ** 9) - datetime.timestamp(gps_date_start)) / (3600*24)
GPS_week = days_epoch//7
time_week = (days_epoch/7  % 1) * (3600*24*7)
obs['GPS_week'] = GPS_week
obs['time_week'] = time_week

### convert column names to developer used column names
nav.columns = ['epoch','SV','af0','af1','af2','iode','crs','dn','m0','cuc','e','cus','sqrta','toe','cic','omg0',
               'cis','i0','crc','omega','odot','idot','CodesL2','GPSWeek','L2Pflag','SVacc','health','TGD','IODC','toc']
nav = nav[['epoch','SV','toc','toe','af0','af1','af2','e','sqrta','dn','m0',
                'omega','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','TGD','IODC','SVacc']]

### edit precise orbit dataframe for easier accessibility
try:
    epoch = pd.to_datetime(p_orb[p_orb.ECEF == 'x']['time'].values)
    sv = p_orb[p_orb.ECEF == 'x']['sv'].values
    x = p_orb[p_orb.ECEF == 'x']['position'].values
    y = p_orb[p_orb.ECEF == 'y']['position'].values
    z = p_orb[p_orb.ECEF == 'z']['position'].values
    
    p_orb2 = pd.DataFrame([epoch,sv,x,y,z]).T
    p_orb2.columns = ["epoch", "SV", "x", "y", "z"]

except:
    p_orb2 = None

### Molave Ref
# ref = ell2cart(np.array([14.6575984,121.0673426,116.7935])).T

### Freshie Walk Ref
# ref = ell2cart(np.array([14.653947053,121.068636975,110.])).T

### CMC Hill Ref
ref = ell2cart(np.array([14.65530195,121.0638391,104.4737])).T

### PTAG Ref
# ref = np.array([-3184320.9618,5291067.5908,1590413.9800])[np.newaxis,:]

### initial state estimate (Single Point Positioning)
# init_x = np.zeros(shape=(8,1))
# P = 5000

#initial position estimate
init_x = np.zeros(shape=(40,1))
P = 500

start = time.time()
count = 0

sample = obs.groupby(time_week)

pos_list, time_list, e_list, n_list, u_list, b_list = [], [], [], [], [], []

sv_list = obs.SV.unique().tolist()
N1_dict = dict(map(lambda x,y: (x,y), list(range(32)),[[0,0] for sv in range(32)]))

for rcvr_time, group in sample:
    # rcvr_time = int(round(rcvr_time))
    # init_x = calc_cp_pos(rcvr_time, nav, group, init_x, N1_dict, N2_dict,N1_list, N2_list, count)
    # N1_list = np.append(N1_list,np.array([list(N1_dict.values())]),axis=0)
    # N2_list = np.append(N2_list,np.array([list(N2_dict.values())]),axis=0)
    
    try:
        # init_x, residuals = calc_rcvr_pos2(rcvr_time,nav,group,init_x,count,freq = 'dual')
        
        # init_x, P, log_likelihood = calc_rcvr_pos3(rcvr_time,
        #                                             nav, group, init_x, P,
        #                                             10, (3e-8)*c, count,freq='single',iono_corr=iono_corr)
        
        # init_x, P, log_likelihood = calc_rcvr_pos3(rcvr_time,
        #                                             nav, group, init_x, P,
        #                                             10, (3e-8)*c, count,freq='dual')
        
        init_x, P, log_likelihood, residuals = calc_cp_pos3(rcvr_time,
                                                    nav, p_orb2, group, init_x, P,
                                                    0.6, (3e-8)*c, N1_dict, count)
        
        # init_x = calc_rcvr_pos4(rcvr_time,nav,group,init_x,P,10.0,count,freq = 'dual')
        # init_x = calc_cp_pos(rcvr_time, nav, group, init_x, N1_dict, N2_dict,N1_list, N2_list, count)
        # N1_list = np.append(N1_list,np.array([list(N1_dict.values())]),axis=0)
        # N2_list = np.append(N2_list,np.array([list(N2_dict.values())]),axis=0)
        
    except Exception as e:
        print(e)
        continue
    except ValueError:
        raise ValueError
    
    x, y, z, b = init_x.flatten()[0], init_x.flatten()[1], init_x.flatten()[2], init_x.flatten()[3]
    lat, lon, height = cart2ell(np.array([[x,y,z]])).flatten()
    b = b/c
    
    print(f"Latitude: {lat}, Longitude: {lon}, Ellipsoidal Height: {height}")
    print(f"Receiver Clock Bias: {b} s")
    print(f"Log-Likelihood: {log_likelihood} \n")
    
    e, n, u = ecef2enu(ref,np.array([[x,y,z]])).flatten()
    
    pos_list.append([lat,lon,height])
    time_list.append(rcvr_time)
    e_list.append(e)
    n_list.append(n)
    u_list.append(u)
    b_list.append(b)
    
    count += 1
    # time.sleep(1)

end = time.time()
print(f"Runtime: {end-start} seconds")

time_list2 = np.array(time_list)
e_list2 = np.array(e_list)
n_list2 = np.array(n_list)
u_list2 = np.array(u_list)
b_list2 = np.array(b_list)

e_list2 = e_list2 - np.median(e_list2)
n_list2 = n_list2 - np.median(n_list2)
u_list2 = u_list2 - np.median(u_list2)
b_list2 = b_list2 - np.median(b_list2)

### basic filtering
condition = np.where((np.abs(e_list2) < 10) & (np.abs(n_list2) < 10) & (np.abs(u_list2) < 10))
time_list2 = time_list2[condition]
e_list2 = e_list2[condition]
n_list2 = n_list2[condition]
u_list2 = u_list2[condition]
b_list2 = b_list2[condition]

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


###plot pseudorange vs time
fig,ax = plt.subplots(figsize=(12,7),dpi=120)
t = obs[obs['SV'] == sv_list[0]]['time_week']
C1C = obs[obs['SV'] == sv_list[0]]['C1C']
L1C = obs[obs['SV'] == sv_list[0]]['L1C'] * l1
ax.scatter(x=t,y=C1C,c = 'black',label = 'C1C measurements')
ax.scatter(x=t,y=L1C,c = 'red',label = 'L1C measurements')
ax.set_title(f'{sv_list[0]}: C1C and L1C Pseudorange measurement vs time')
ax.legend()

###plot pseudorange vs time
# fig,ax = plt.subplots(figsize=(12,7),dpi=120)
# t = obs[obs['SV'] == sv_list[0]]['time_week']
# L1C = obs[obs['SV'] == sv_list[0]]['L1C'] * l1
# L2W = obs[obs['SV'] == sv_list[0]]['L2W'] * l2
# C1C = obs[obs['SV'] == sv_list[0]]['C1C']
# C2W = obs[obs['SV'] == sv_list[0]]['C2W']
# ax.scatter(x=t,y=L1C-L2W,c = 'black',label = 'L1-L2 Geom-free measurements')
# ax.scatter(x=t,y=C2W-C1C,c = 'red',label = 'C2-C1 Geom-free measurements')
# ax.set_title(f'{sv_list[0]}: Geom-free combinations of C2-C1 and L1-L2 vs time')
# ax.legend()

###plot each component versus time 
fig,ax = plt.subplots(nrows=4,figsize=(12,18),dpi=120)
ax[0].scatter(x=time_list2,y=e_list2,c = 'black',label = 'Easting')
ax[0].set_title(f'Easting vs time, Mean: {e_list2.mean()} Standard Deviation: {e_list2.std()}')
ax[0].grid(True)
ax[0].legend()


ax[1].scatter(x=time_list2,y=n_list2,c = 'red',label = 'Northing')
ax[1].set_title(f'Northing vs time, Mean: {n_list2.mean()} Standard Deviation: {n_list2.std()}')
ax[1].grid(True)
ax[1].legend()


ax[2].scatter(x=time_list2,y=u_list2,c = 'blue',label = 'Up')
ax[2].set_title(f'Up vs time, Mean: {u_list2.mean()} Standard Deviation: {u_list2.std()}')
ax[2].grid(True)
ax[2].legend()


ax[3].scatter(x=time_list2,y=b_list2,c = 'green',label = 'User Clock Bias')
ax[3].set_title(f'Receiver Clock Bias vs time, Mean: {b_list2.mean()} Standard Deviation: {b_list2.std()}')
ax[3].grid(True)
ax[3].legend()
plt.show()