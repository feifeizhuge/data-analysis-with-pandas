#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:18:30 2019

@author: hechen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(__file__)
csv_path = current_path + '/2016_02_10_0000_1687_export.csv'
Data = pd.read_csv(csv_path, ';')


"""
['Unix Timestamp [ms]', 
'Time [Europe/Amsterdam]',
       'Battery - Avg cell voltage [mV]', 
       'Battery - Avg cell temp [°C]',
       'Battery - Max. charge current [A]',
       'Battery - Max. discharge current [A]', 
       'Battery - Cell nr. max volt',
       'Battery - Max cell voltage [mV]', 
       'Battery - Cell nr. min volt',
       'Battery - Min cell voltage [mV]', 
       'Battery - Max. charge voltage [V]',
       'Battery - Current [A]', 
       'Battery - Battery power [kW]',
       'Battery - Voltage [V]', 
       'Battery - State of Charge [%]',
       'Battery - State of Health [%]', 
       'Primove - Charger wayside present',
       'Primove - Charging state', 
       'Primove - Charging current [A]',
       'Primove - Pick-up temp [°C]', 
       'Primove - Rectrifier temp [°C]',
       'Primove - Charging voltage [V]', 
       'Primove - Pick-up position [mm]',
       'Primove - Pick-up position control', 
       'Battery - TCU compressor status',
       'Battery - TCU heater status',
       'Vehicle - Accelerator pedal position [%]',
       'Vehicle - Accelerator pedal switch',
       'Engine - Actual Engine Torque [%]',
       'Vehicle - Ambient Air Temperature [°C]',
       'Air pressure - Pressure front axle left [bar]',
       'Air pressure - Pressure front axle right [bar]',
       'Air pressure - Pressure rear axle left [bar]',
       'Air pressure - Pressure rear axle right [bar]',
       'Air pressure - Pressure brake circuit 1 [bar]',
       'Air pressure - Pressure brake circuit 2 [bar]',
       'Vehicle - Brake pedal switch', 'Vehicle - Current Gear',
       'Engine - Driver Engine Torque demand [%]',
       'Doors - Enable status Doors 1', 'Doors - Enable status Doors 2',
       'Doors - Enable status Doors 3', 'Doors - Enable status Doors 4',
       'Doors - Enable status Doors 5', 'Engine - Motor Speed RPM [RPM]',
       'Vehicle - High resolution vehicle distance [km]',
       'Doors - Lock status Doors 1', 'Doors - Lock status Doors 2',
       'Doors - Lock status Doors 3', 'Doors - Lock status Doors 4',
       'Doors - Lock status Doors 5', 'Doors - Open status Doors 1',
       'Doors - Open status Doors 2', 'Doors - Open status Doors 3',
       'Doors - Open status Doors 4', 'Doors - Open status Doors 5',
       'Doors - Positions Doors', 'Doors - Status Doors',
       'Vehicle - Tachograph Speed [km/h]',
       'Vehicle - Wheel based Speed [km/h]', 'IVH - Altitude [m]',
       'IVH - GPS Course [°]', 'IVH - GPS position', 'IVH - GPS speed [km/h]',
       'IVH - Satellites', 'IVH - 24V Battery [V]', 'IVH - Online status',
       'HVAC - AC compressor state [%]', 'Auxiliary - Air compressor state',
       'Vehicle - Brake pedal position [%]', 'HVAC - Cabin air temp [°C]',
       'HVAC - Cabin air temp setpoint [°C]', 'HVAC - Condenser fan state',
       'HVAC - Evaporator fan state', 'HVAC - Floor air heater state',
       'Vehicle - Ignition State',
       'HVAC - Outlet air temperature floor unit [°C]',
       'HVAC - Outlet air temperature roof unit [°C]',
       'HVAC - Recirculation Air flap position [%]',
       'Auxiliary - Steering pump state',
       'HVAC - Water heater outlet temp [°C]',
       'Auxiliary - Total Power 24V [kW]', 'Auxiliary - Total Power HV [kW]',
       'Powertrain - Inverter temperature [°C]',
       'Powertrain - Motor temperatur status',
       'Powertrain - Traction Power [kW]', 'Vehicle - Vehicle state'],
      dtype='object')
"""

#%% obversation
# Modifications to the data or indices of the copy will 
# not be reflected in the original object 
X = Data.copy(deep=True)

# feststellen simpling frequency
print(X['Time [Europe/Amsterdam]'][0:40])

#%%

# slice operation, simple every 20 points
X = X[0::20]

'''
print(X['Time [Europe/Amsterdam]'][0:40])
X = X[0::20]
print(X['Time [Europe/Amsterdam]'][0:50])
X['Unix Timestamp [ms]'][0:20]
sjdada
X['Unix Timestamp [ms]'][0:20]
pd.to_datetime(1455072686227)
pd.to_datetime(1455072686)
pd.to_datetime(1455072686227,unit='s')
pd.to_datetime(1455072686227,unit='ms')
X['Unix Timestamp [ms]'][0:20]
pd.to_datetime(1455072686227,unit='ms')
print(X['Time [Europe/Amsterdam]'][0:50])
pd.to_datetime(1455072686227,unit='ms')
pd.to_datetime(1455072686227/1000,unit='s')
1455072686227/1000
int(1455072686227/1000)
pd.to_datetime(1455072686227//1000,unit='s')
pd.to_datetime(X['Unix Timestamp [ms]'][0:30]//1000,unit='s')
'''

#%%
#time = Data['seconds']
#time = (time-time.iloc[0]) / 60
#my_x_ticks = np.arange(time.iloc[0], time.iloc[-1], 1)
#
#plt.plot(time, Data['GPS Speed'])
#plt.grid(axis='x')
#plt.xticks(my_x_ticks)




