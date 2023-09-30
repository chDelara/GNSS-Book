# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:06:19 2023

@author: ASTI
"""

import numpy as np
from scipy.constants import c
import time


rwy30Start = np.array([[-2694685.473,-4293642.366,3857878.924]])
rwy30End = np.array([[-2694892.460,-4293083.225,3858353.437]])


def ell2cart(point):
    """
    

    Parameters
    ----------
    point : numpy.array
        an array containing coordinates of a point in WGS84 coordinates.

    Returns
    -------
    cart_point : numpy.array
        an array containing transformed coordinates from WGS84 to ECEF coordinates

    """
    
    lat, lon, h = point.flatten()
    
    # WGS84 parameters
    a = 6378137.0 # semi-major axis
    f = 1/298.257223563 # inverse flattening
    b = a - (a*f) # semi-minor axis
    e = np.sqrt(2*f - f**2)
    omega_E = 7292115.0e-11 # Earth's angular velocity
    GM = 3986004.418e8 # Earth's gravitational constant
    
    
    N = a / np.sqrt(1 - (e**2) * np.sin(np.radians(lat))**2)
    
    cart_point = [ [(N + h) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))],
                  [(N + h) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))],
                  [((N*(1-e**2)) + h) * np.sin(np.radians(lat))] ]
    
    return np.array(cart_point)

def cart2ell(point):
    # WGS84 parameters
    a = 6378137.0 # semi-major axis
    f = 1/298.257223563 # inverse flattening
    b = a - (a*f) # semi-minor axis
    e = np.sqrt(2*f - f**2)
    omega_E = 7292115.0e-11 # Earth's angular velocity
    GM = 3986004.418e8 # Earth's gravitational constant
    
    x, y, z = point.flatten()
    lon = np.degrees(np.arctan2(y,x))
    
    p = np.sqrt(x**2 + y**2)
    lat0 = np.arctan(z/((1-e**2)*p))
    N = a / (np.sqrt(1 - ((e**2) * np.sin(lat0)**2)))
    h0 = (p/np.cos(lat0)) - N
    
    delta_lat = 1000.0
    delta_h = 1000.0

    while (delta_lat >= 1e-7) | (delta_h >= 1e-4):
        N_d = np.sqrt(1 - ((e**2) * np.sin(lat0)**2))
        N = a / N_d
        
        h1 = (p/np.cos(lat0)) - N
        lat1 = np.arctan( z/ (p * (1 - (e**2 * (N/(N+h1))))) )
        
        delta_lat = abs(lat1 - lat0)
        delta_h = abs(h1 - h0)
        
        lat0, h0 = lat1, h1

    return np.array([[np.degrees(lat0),lon,h0]])

def ecef2enu(point_origin,point):
    lat, lon, h = cart2ell(point_origin).flatten()
    rot_mat = np.array([[-np.sin(np.radians(lon)),np.cos(np.radians(lon)),0],
                        [-np.sin(np.radians(lat))*np.cos(np.radians(lon)),-np.sin(np.radians(lat))*np.sin(np.radians(lon)),np.cos(np.radians(lat))],
                        [np.cos(np.radians(lat))*np.cos(np.radians(lon)),np.cos(np.radians(lat))*np.sin(np.radians(lon)),np.sin(np.radians(lat))]])
    
    return (rot_mat @ point.T) - (rot_mat @ point_origin.T)

def calc_az_el(point):
    e,n,u = point.flatten()
    dist = np.sqrt(np.square(point).sum())
    
    az = np.degrees(np.arctan2(e,n))
    el = np.degrees(np.arcsin(u/dist))
    
    return np.array([[az],
                    [el]])