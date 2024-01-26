# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:20:32 2024

@author: ASTI
"""

import georinex as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

gps_date_start = datetime(1980,1,6)

def readObs(file):
    obs = gr.load(file,use="G").to_dataframe().reset_index(drop=False)
    obs_columns = obs.columns.tolist()
    obs_columns[:2] = ['SV','epoch']
    obs.columns = obs_columns

    days_epoch = (((obs.epoch - timedelta(hours=8)).values.astype(np.int64) // 10 ** 9) - datetime.timestamp(gps_date_start)) / (3600*24)
    GPS_week = days_epoch//7
    time_week = (days_epoch/7  % 1) * (3600*24*7)
    obs['GPS_week'] = GPS_week
    obs['time_week'] = time_week
    
    return obs
    
def readNav(file):
    nav = gr.load(file,use="G")
    iono_corr = None
    try:
        iono_corr = nav.ionospheric_corr_GPS
    except:
        pass
    
    nav = nav.to_dataframe().reset_index(drop=False)
    nav.columns = ['epoch','SV','af0','af1','af2','iode','crs','dn','m0','cuc','e','cus','sqrta','toe','cic','omg0',
                   'cis','i0','crc','omega','odot','idot','CodesL2','GPSWeek','L2Pflag','SVacc','health','TGD','IODC','toc']
    nav = nav[['epoch','SV','toc','toe','af0','af1','af2','e','sqrta','dn','m0',
                    'omega','omg0','i0','odot','idot','cus','cuc','cis','cic','crs','crc','IODC']]
    
    return nav, iono_corr