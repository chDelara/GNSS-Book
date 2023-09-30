# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:23:09 2023

@author: Cholo
"""

import numpy as np
import matplotlib.pyplot as plt

def shift_register(register,feedback,output):
    """

    Parameters
    ----------
    register : integer
        The length of the shift register.
    feedback : list [int,int]
        A list containing the position values for a particular SV for feedback.
    output : list or 
        DESCRIPTION.

    Returns
    -------
    out : ndarray.

    """
    
    # calculate the output
    out = [register[position-1] for position in output]
    
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]
        
    # module 2 feedback computation
    fb = sum([register[position-1] for position in feedback]) % 2
    
    # shift register values to the right by 1 cell
    register = np.roll(register,1)
    
    # put feedback to position 1
    register[0] = fb
    
    return register,out
    
def PRN_Gen(sv,G1=np.ones(1),G2=np.ones(1)):
    SV = {
       1: [2,6],
       2: [3,7],
       3: [4,8],
       4: [5,9],
       5: [1,9],
       6: [2,10],
       7: [1,8],
       8: [2,9],
       9: [3,10],
      10: [2,3],
      11: [3,4],
      12: [5,6],
      13: [6,7],
      14: [7,8],
      15: [8,9],
      16: [9,10],
      17: [1,4],
      18: [2,5],
      19: [3,6],
      20: [4,7],
      21: [5,8],
      22: [6,9],
      23: [1,3],
      24: [4,6],
      25: [5,7],
      26: [6,8],
      27: [7,9],
      28: [8,10],
      29: [1,6],
      30: [2,7],
      31: [3,8],
      32: [4,9],
    }
    
    if (len(G1) == 1) | (len(G2) == 1):
        G1 = np.ones(10,dtype=int)
        G2 = np.ones(10,dtype=int)
    else:
        pass
    
    ca_code = []
    
    for chip in range(1023):
        G1,g1 = shift_register(G1,[3,10],[10])
        G2,g2 = shift_register(G2,[2,3,6,8,9,10],SV[sv])
        
        ca = (g1 + g2) % 2
        ca_code.append(ca)
        
    return np.array(ca_code)


#####--- Generate PRN Code ---#####
ca = PRN_Gen(19)
ca2 = -2*ca+1
ca3 = np.roll(ca2,200)

corr_arr = []
for shift in np.arange(-1023,1023):
    corr = np.sum(ca3 * np.roll(ca2,shift))/1023
    corr_arr.append(corr)
    
fig = plt.figure(figsize=(12,6))
plt.plot(corr_arr)
plt.title('SV 19 PRN Code Correlation with 200 chip delay')
plt.show()


####--- Compare SV 19 and SV 25 PRN Code ---####
ca19 = PRN_Gen(19)
ca19 = -2*ca19+1

ca25 = PRN_Gen(25)
ca25 = -2*ca25+1

corr_arr = []
for shift in np.arange(-1023,1023):
    corr = np.sum(ca19 * np.roll(ca25,shift))/1023
    corr_arr.append(corr)

fig = plt.figure(figsize=(12,6))
plt.plot(corr_arr)
plt.title('SV 19 PRN Code Correlation SV 25')
plt.show()

####--- Compare SV 19 and SV 5 PRN Code ---####
ca19 = PRN_Gen(19)
ca19 = -2*ca19+1

ca5 = PRN_Gen(5)
ca5 = -2*ca5+1

corr_arr = []
for shift in np.arange(-1023,1023):
    corr = np.sum(ca19 * np.roll(ca5,shift))/1023
    corr_arr.append(corr)

fig = plt.figure(figsize=(12,6))
plt.plot(corr_arr)
plt.title('SV 19 PRN Code Correlation SV 5')
plt.show()

####--- Compare SV 19 and SV 5, SV 25, 5 PRN Code ---####
ca19 = PRN_Gen(19)
ca19 = -2*ca19+1
ca19_2 = np.roll(ca19,200)

ca25 = PRN_Gen(25)
ca25 = -2*ca25+1
ca25 = np.roll(ca25,905)

ca5 = PRN_Gen(5)
ca5 = -2*ca5+1
ca5 = np.roll(ca5,75)

ca_3SV = ca19_2+ca25+ca5

corr_arr = []
for shift in np.arange(-1023,1023):
    corr = np.sum(ca_3SV * np.roll(ca19,shift))/1023
    corr_arr.append(corr)

fig = plt.figure(figsize=(12,6))
plt.plot(corr_arr)
plt.title('PRN Code Correlation with delays(chips): SV 19: +200, SV25: +905, SV5: +75')
plt.show()


####--- simulation of random noise and signals from other SVs ---####
ca19 = PRN_Gen(19)
ca19 = -2*ca19+1
ca19_2 = np.roll(ca19,200)

ca25 = PRN_Gen(25)
ca25 = -2*ca25+1
ca25 = np.roll(ca25,905)

ca5 = PRN_Gen(5)
ca5 = -2*ca5+1
ca5 = np.roll(ca5,75)

noise = np.random.normal(loc=0,scale=4,size=1023)

ca_3SV = ca19_2 + ca25 + ca5 + noise

corr_arr = []
for shift in np.arange(-1023,1023):
    corr = np.sum(ca_3SV * np.roll(ca19,shift))/1023
    corr_arr.append(corr)
    
fig = plt.figure(figsize=(12,6))
plt.plot(corr_arr)
plt.title('SV 19 PRN Code Correlation simulated random noise and other SV signals')
plt.show()

fig = plt.figure(figsize=(12,6))
plt.plot(np.arange(0,1023),ca_3SV)
plt.title('Simulated signal from SV 19, SV25, SV 5 PRN Codes and noise signal')
plt.show()

####--- plot each PRN codes and noise ---####
ca19 = PRN_Gen(19)
ca19 = -2*ca19+1
ca19_2 = np.roll(ca19,200)

ca25 = PRN_Gen(25)
ca25 = -2*ca25+1
ca25 = np.roll(ca25,905)

ca5 = PRN_Gen(5)
ca5 = -2*ca5+1
ca5 = np.roll(ca5,75)

noise = np.random.normal(loc=0,scale=4,size=1023)

fig, ax = plt.subplots(4,1,figsize=(20,12))

ax[0].plot(np.arange(0,1023),ca19_2)
ax[0].set(title='SV 19 PRN Code')

ax[1].plot(np.arange(0,1023),ca5)
ax[1].set(title='SV 25 PRN Code')

ax[2].plot(np.arange(0,1023),ca25)
ax[2].set(title='SV 5 PRN Code signal')

ax[3].plot(np.arange(0,1023),noise)
ax[3].set(title='Noise signal')

fig.tight_layout()
plt.show()