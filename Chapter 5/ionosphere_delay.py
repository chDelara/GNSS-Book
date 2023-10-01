# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:06:37 2023

@author: Cholo
"""

import pandas as pd
from datetime import datetime, timedelta
import time

gps_date_start = datetime(1980,1,6)
f1 = 1575.42 #MHz
f2 = 1227.6 #MHz
f5 = 1176. #MHz

rio = pd.read_fwf(r'D:/Cholo/Self-Reading(Geodesy)/Books/GPSBookCD/Data/Pigeon_Point/September_18_2000/Parsed/pp091800.rio',header=None)
rio.columns = ["GPS_week","rcvr_tow","sv","C1","L1","L2","P1","P2","D1","D2"]

###calculating GPS date for each measurement 
gps_date_list = []
for idx, row in rio.iterrows():
    week = row.GPS_week
    sec = row.rcvr_tow
    gps_date = gps_date_start + timedelta(weeks=week) + timedelta(seconds=sec)
    gps_date_list.append(gps_date)
end = time.time()

rio.insert(loc=0,column="GPS_date",value=gps_date_list)

io_delay = ((f2**2) / ((f1**2) + (f2**2))) * (rio.P2 - rio.P1)