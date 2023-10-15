# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 00:35:20 2023

@author: Cholo
"""

import numpy as np
import numpy.linalg as la

x1 = np.array([[3.,9.53]])
x2 = np.array([[8.41,8.65]])
x3 = np.array([[-1.23,6.99]])

H = np.append(x1,np.append(x2,x3,axis=0),axis=0)

p1 = 1.19
p2 = 3.71
p3 = -1.71

p_arr = np.array([[p1,p2,p3]]).T

init_x = np.array([0,0,0])

# x, y, b = [0,0,0]

# d1 = np.sqrt(np.sum(np.square(x1 - [x,y])))
# d2 = np.sqrt(np.sum(np.square(x2 - [x,y])))
# d3 = np.sqrt(np.sum(np.square(x3 - [x,y])))

# d_arr = np.array([[d1],[d2],[d3]])

# G = -(G-[x,y])/d_arr
# G = np.append(G,np.array([[1,1,1]]).T,axis=1)

def G_mat(init_x, G):
    x, y, b = init_x
    
    d_arr = np.sqrt(np.sum(np.square(G - [x,y]),axis=1)[:,np.newaxis])

    G = -(G-[x,y])/d_arr
    G = np.append(G,np.array([[1,1,1]]).T,axis=1)
    
    return G
    

def est_p(init_x,G):
    return np.sqrt(np.sum(np.square(G - init_x[:2]),axis=1)[:,np.newaxis]) + init_x[-1]

delta_xb = la.inv(G_mat(init_x,H)) @ (p_arr - est_p(init_x,H))

while ~(np.abs(delta_xb) < 1e-6).all():
    init_x = init_x + delta_xb.flatten()
    delta_xb = la.inv(G_mat(init_x,H)) @ (p_arr - est_p(init_x,H))
    print(init_x)