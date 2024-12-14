# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:38:15 2024

@author: ASTI
"""

import numpy as np

def saastamoinen(height, elevation):
    
    h_rel = 70
    
    if height > 1000:
        height = 100
    else:
        pass
    
    pressure = 1013.25 * np.power(1 - ((2.2557e-5) * height),5.2568) # total pressure
    abs_temp = 15.0 - ((6.5e-3) * height) + 273.15 # absolute temperature
    part_press = 6.108 * np.exp((17.15*abs_temp - 4648)/(abs_temp - 38.48)) * (h_rel/100) # partial pressure
    
    zenith = np.pi/2 - np.radians(elevation)
    
    Tr = (0.002277/np.cos(zenith)) * (pressure + ((1255/abs_temp + 0.05) * part_press) - np.square(np.tan(zenith)) )
    
    return Tr