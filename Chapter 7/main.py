# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:22:45 2023

@author: ASTI
"""

import georinex as gr
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from methods.sat_pos import *
from methods.localization import *
from datetime import datetime,timedelta
import time

gps_date_start = datetime(1980,1,6)
f1 = 1575.42 #MHz
f2 = 1227.6 #MHz
f5 = 1176. #MHz

# obs = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 1/Molave/Molave/IGS000USA_R_20193020215_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'D:/Cholo/UP/5th Year - 1st Sem - BS Geodetic Engineering/GE 155.1/GNSS/Day 1/Molave/Molave/IGS000USA_R_20193020215_00M_01S_MN.rnx',use="G")

obs = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/IGS000USA_R_20193020215_00M_01S_MO.rnx',use="G").to_dataframe().reset_index(drop=False)
nav = gr.load(r'C:/Users/ASTI/Desktop/GNSS/UP data/IGS000USA_R_20193020215_00M_01S_MN.rnx',use="G")

# obs = gr.load(r'C:/Users/ASTI/Desktop/GNSS/PTAG00PHL_R_20230180100_01H_30S_MO.crx',use="G").to_dataframe().reset_index(drop=False)
# nav = gr.load(r'C:/Users/ASTI/Desktop/GNSS/PTAG00PHL_R_20230180000_01H_GN.rnx',use="G")

iono_corr = nav.ionospheric_corr_GPS
nav = nav.to_dataframe().reset_index(drop=False)

obs_columns = obs.columns.tolist()
obs_columns[:2] = ['SV','epoch']
obs.columns = obs_columns

days_epoch = (((obs.epoch - timedelta(hours=8)).values.astype(np.int64) // 10 ** 9) - datetime.timestamp(gps_date_start)) / (3600*24)
GPS_week = days_epoch//7
time_week = (days_epoch/7  % 1) * (3600*24*7)
obs['GPS_week'] = GPS_week
obs['time_week'] = time_week



nav.columns = ['epoch','SV','af0','af1','af2','iode','crs','dn','m0','cuc','e','cus','sqrta','toe','cic','omg0',
               'cis','i0','crc','omega','odot','idot','CodesL2','GPSWeek','L2Pflag','SVacc','health','TGD','IODC','toc']
nav = nav[['epoch','SV','toc','toe','af0','af1','af2','e','sqrta','dn','m0',
                'omega','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','IODC']]


