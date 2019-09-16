#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:18:30 2019

@author: hechen
"""

import pandas as pd
import os

current_path = os.path.dirname(__file__)
data_path = '2016_02_12/'
data_name = '2016_02_12_0000_1685_export'
csv_path = current_path + '/' + data_path + data_name + '.csv'
Data = pd.read_csv(csv_path, ';')
print('Data loaded successfully')


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
#print(X['Time [Europe/Amsterdam]'][0:40])

#%%

# 降采样，每20个点采集一次数据，降低数据量
#X = X[0::20]

# 把时间精度从毫秒[ms]改成秒[s]
X['Unix Timestamp [ms]'] = X['Unix Timestamp [ms]'].apply(lambda x: x//1000+60*60)
# 把修改后的这一列修改名字
X.rename(columns={'Unix Timestamp [ms]': 'Unix Timestamp [s]'},inplace=True)
# 把对应的列['Time [Europe/Amsterdam]']按照新的时间戳转换
X['Time [Europe/Amsterdam]'] = pd.to_datetime(X['Unix Timestamp [s]'],unit='s')

# 把['Time [Europe/Amsterdam]']上重复的项删除
X.drop_duplicates('Time [Europe/Amsterdam]','first',inplace=True)

# 特定处理12-02-1685的数据
#X.reset_index(drop=True, inplace=True)
#X.loc[58880:58940,'Vehicle - Tachograph Speed [km/h]'] = 0

savepath = data_path + 'processed data/' + data_name.replace('export','dowmsampling') + '.csv'
X.to_csv(savepath,index=False,sep=';')
#X.to_csv("2016_02_25_0000_1685_with_index.csv",index=True,sep=';')
print('Data saved successfully')




#%%
#time = Data['seconds']
#time = (time-time.iloc[0]) / 60
#my_x_ticks = np.arange(time.iloc[0], time.iloc[-1], 1)
#
#plt.plot(time, Data['GPS Speed'])
#plt.grid(axis='x')
#plt.xticks(my_x_ticks)




