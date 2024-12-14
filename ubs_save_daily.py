# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:57:34 2024

@author: ASTI
"""

import os, sys, time, serial, csv, math, ast, signal
from datetime import datetime
from io import BytesIO
import pynmea2

### set-up Keyboard Interrupt signal from Unix terminal
def handler(signum, frame):
	response = input("\nEnd GPS logging? y/n\n")
	if response == 'y':
		raise KeyboardInterrupt
	else:
		print("Continuing Logging\n")

### Set folder name for output
year = datetime.now().year
folder_name = f"sagap{year}/gps_bin"

###Set paths for port and GPS output
port = "/dev/ttyACM0"
logfolder = f"/home/bry/Desktop/{folder_name}"

start = time.time()
serialPort = serial.Serial(port, baudrate = 9600, timeout = 0.5)

if not os.path.exists(logfolder):
	os.makedirs(logfolder)
else:
	pass

signal.signal(signal.SIGINT, handler)

while True:
    row_gps = []
    ### Set Output directory and filename
    record_time = datetime.now()
    filename = f"gps_{record_time.strftime('%Y%m%d %H_%M_%S')}.ubx"
    file_path = os.path.join(logfolder, filename)
    
    with open(file_path, "ab") as ubxfile:
        current_time = datetime.now()
        while True:
            if current_time.day > record_time.day:
                break
            try:
                data_time = time.strftime("%Y-%m-%d %H:%M:%S")
                string_gps = serialPort.readline()
                row_gps.append(string_gps)
      
                if str(string_gps).find('GGA') > 0:
                    msg = pynmea2.parse(string_gps.decode("utf-8"))
                    latddm = float(msg.lat)
                    londdm = float(msg.lon)
                    altitude = float(msg.altitude)
                    num_sats = int(msg.num_sats)
                    hdop = float(msg.horizontal_dil)
      
                    latdd = math.floor(latddm/100) + (latddm - math.floor(latddm/100)*100)/60
                    londd = math.floor(londdm/100) + (londdm - math.floor(londdm/100)*100)/60
      
                    print(f"Datetime: {data_time}")
                    print(f"Latitude: {latdd}")
                    print(f"Longitude: {londd}")
                    print(f"Altitude: {altitude}")
                    print(f"Number of Satellites: {num_sats}")
                    print(f"HDOP: {hdop}\n")
                
                ubxfile.write(BytesIO(string_gps).getbuffer())
       
            except KeyboardInterrupt:
                raise Exception

            except UnicodeDecodeError:
                continue
    
    serialPort.close()