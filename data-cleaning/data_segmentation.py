#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:17:34 2019

@author: hechen
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import sklearn
import math
import itertools

#%% Data cleaning
# loading dataset
# 目前可用的数据日期：02_11_1685 02_12_1685
current_path = os.path.dirname(__file__)
data_path = '/2016_02_11/processed data/'
data_name = '2016_02_11_0000_1685_dowmsampling.csv'
csv_path = current_path + data_path + data_name
Data = pd.read_csv(csv_path, ';')

# make a copy of data 
X = Data.copy(deep=True)
X.info()
X.drop(X.index[0], inplace=True)
X.reset_index(drop=True, inplace=True)

# convert the all columns to a list
Features_name = X.columns.values

# print name of feature, which doesn't have value(Nan)
Feature_Nan = Features_name[X.isnull().all().values]


# print name of feature, which have value
Feature_valid = Features_name[~X.isnull().all().values]

#Feature_valid_zero = Features_name[np.logical_and(~X.isnull().all().values,(X != 0).all().values)]
#Feature_valid_Nozero = Features_name[np.logical_and(~X.isnull().all().values,~(X != 0).all().values)]

# drop the all columns which value are all Nan
X.dropna(axis=1, how='all', inplace=True)

# replace the Nan value with naught, because Nan value means no meansure happend
X.fillna(value=0, inplace=True)


#%% Data visulization
'''
['Unix Timestamp [s]', 'Time [Europe/Amsterdam]',
       'Battery - Avg cell voltage [mV]',
       'Battery - Max. charge current [A]', 
       'Battery - Cell nr. max volt',
       'Battery - Max cell voltage [mV]', 
       'Battery - Cell nr. min volt',
       'Battery - Min cell voltage [mV]', 
       'Battery - Current [A]',
       'Battery - Battery power [kW]', 
       'Battery - Voltage [V]',
       'Battery - State of Charge [%]',
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
       'Engine - Actual Engine Torque [%]',
       'Vehicle - Ambient Air Temperature [°C]', 
       'Vehicle - Current Gear',
       'Engine - Motor Speed RPM [RPM]',
       'Vehicle - High resolution vehicle distance [km]',
       'Vehicle - Tachograph Speed [km/h]', 
       'IVH - Altitude [m]',
       'IVH - GPS position', 
       'IVH - GPS speed [km/h]', 
       'IVH - Satellites',
       'IVH - 24V Battery [V]', 
       'IVH - Online status',
       'HVAC - AC compressor state [%]',
       'Auxiliary - Air compressor state',
       'Vehicle - Brake pedal position [%]', 
       'HVAC - Cabin air temp [°C]',
       'HVAC - Cabin air temp setpoint [°C]',
       'HVAC - Condenser fan state', 
       'HVAC - Evaporator fan state',
       'Vehicle - Ignition State',
       'HVAC - Recirculation Air flap position [%]',
       'Auxiliary - Steering pump state',
       'HVAC - Water heater outlet temp [°C]',
       'Auxiliary - Total Power 24V [kW]',
       'Auxiliary - Total Power HV [kW]',
       'Powertrain - Inverter temperature [°C]',
       'Powertrain - Motor temperatur status',
       'Powertrain - Traction Power [kW]', 
       'Vehicle - Vehicle state']
'''
# plot the first battery values
colorset = itertools.cycle(["#1f77b4", "#ff7f0e", "#2ca02c",
     "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

battery = [
           'Vehicle - Tachograph Speed [km/h]',
           'Vehicle - Vehicle state',
           'Battery - Avg cell voltage [mV]',
           'Battery - Max. charge current [A]', 
           'Battery - Cell nr. max volt',
           'Battery - Max cell voltage [mV]', 
           'Battery - Cell nr. min volt',
           'Battery - Min cell voltage [mV]', 
           'Battery - Current [A]',
           'Battery - Battery power [kW]', 
           'Battery - Voltage [V]',
           'Battery - State of Charge [%]' 
            ]

primove = [
           'Vehicle - Tachograph Speed [km/h]',
           'Primove - Charger wayside present', 
           'Primove - Charging state',
           'Primove - Charging current [A]', 
           'Primove - Pick-up temp [°C]',
           'Primove - Rectrifier temp [°C]', 
           'Primove - Charging voltage [V]',
           'Primove - Pick-up position [mm]',
           'Primove - Pick-up position control'
        ]


vihicle1 = [
       'Vehicle - Tachograph Speed [km/h]',
       'Battery - TCU compressor status', 
       'Battery - TCU heater status',
       'Engine - Actual Engine Torque [%]',
       'Vehicle - Ambient Air Temperature [°C]', 
       'Vehicle - Current Gear',
       'Engine - Motor Speed RPM [RPM]',
       'Vehicle - High resolution vehicle distance [km]',
       'Vehicle - Tachograph Speed [km/h]'
        ]


IVH = [
       'Vehicle - Tachograph Speed [km/h]',
       'IVH - 24V Battery [V]', 
       'IVH - Online status',
       'HVAC - AC compressor state [%]',
       'Auxiliary - Air compressor state',
       ]


HVAC = [
       'Vehicle - Tachograph Speed [km/h]',
       'Vehicle - Brake pedal position [%]',
       'Vehicle - Accelerator pedal position [%]',
       'HVAC - Cabin air temp [°C]',
       'HVAC - Cabin air temp setpoint [°C]',
       'HVAC - Condenser fan state', 
       'HVAC - Evaporator fan state',
       'Vehicle - Ignition State',
       'HVAC - Recirculation Air flap position [%]',
       'Auxiliary - Steering pump state',
       'HVAC - Water heater outlet temp [°C]'
        ]

powertrain = [
       'Vehicle - Tachograph Speed [km/h]',
       'Auxiliary - Total Power 24V [kW]',
       'Auxiliary - Total Power HV [kW]',
       'Powertrain - Inverter temperature [°C]',
       'Powertrain - Motor temperatur status',
       'Powertrain - Traction Power [kW]', 
       'Vehicle - Vehicle state'
        ]

def plot_feature(data, feature_list):
    f, ax = plt.subplots(len(feature_list),1,sharex=True)
    for ind in range(len(ax)):
        
        ax[ind].plot(data.index,data[feature_list[ind]], label=feature_list[ind], color=next(colorset))
        ax[ind].legend(loc='center right')
    
plot_feature(X, battery)
#plot_feature(X, primove)
#plot_feature(X, vihicle1)
#plot_feature(X, IVH)
#plot_feature(X, HVAC)
#plot_feature(X,powertrain)


# ***** convoriance matrix *****
#plt.figure()
#selection = ['Vehicle - Tachograph Speed [km/h]',
#             'Battery - Current [A]',     
#            'Battery - Battery power [kW]', 
#            'Battery - Voltage [V]',
#            'Battery - State of Charge [%]']
#
#corr_matrix = X[selection][20000:25000].corr()
#sns.heatmap(corr_matrix, center=0, annot=True)

#%% Data segmentation

first_nonzero_index = X['Vehicle - Tachograph Speed [km/h]'].nonzero()[0][0]
last_nonzero_index = X['Vehicle - Tachograph Speed [km/h]'].nonzero()[0][-1]

## ***** visulization result of kickoff head and tail zero data *****
#plt.figure()
#plt.plot(X['Vehicle - Tachograph Speed [km/h]'])
#plt.vlines(first_nonzero_index, -5, 60, linestyles='dashdot', linewidth=2)
#plt.vlines(last_nonzero_index, -5, 60, linestyles='dashdot', linewidth=2)

## test median filter for control signal
#median_filter = signal.medfilt(X['Primove - Pick-up position control'], 5)
## plot filtered signal
#f, ax = plt.subplots(2,1,sharex=True)
#ax[0].plot(X.index,median_filter, label='median_filter', color=colorset[0])
#ax[1].plot(X.index,X['Primove - Pick-up position control'], label='median_filter', color=colorset[0])

# ***** using slice operation to segmentation data *****
X = X[first_nonzero_index:last_nonzero_index]
# smooth the pick-up position control signal
X['Primove - Pick-up position control'] = signal.medfilt(X['Primove - Pick-up position control'], 5)
X.reset_index(drop=True, inplace=True)

plt.figure()
plt.plot(X['Vehicle - Tachograph Speed [km/h]'])
plt.plot(X['Primove - Pick-up position control'] * 30)
plt.plot(X['Primove - Pick-up position control'].rolling(2,min_periods = 1).mean() * 70,'.')
plt.legend(['Tachograph Speed [km/h]','Primove - Pick-up position control','Mean of Pick-up position control'])

X['Primove - Pick-up position control'].rolling(2,min_periods = 1).mean()

def zero_edge_detection(signal, rolling_size):
    
    mean_value = signal.rolling(rolling_size, min_periods = 1).mean()
    step_index = np.nonzero((mean_value != 0) & (mean_value !=1))[0].tolist()
    
    return step_index

step_signal = zero_edge_detection(X['Primove - Pick-up position control'], 2)
# add head element to list
step_signal.insert(0,X.index[0])
# add tail element to list
step_signal.append(X.index[-1])

segmentation_list = []
segmentation_list_1 = []
head_list = step_signal[0::2]
tail_list = step_signal[1::2]
final_feature = ['Unix Timestamp [s]', 
                 'Time [Europe/Amsterdam]',
                 'Vehicle - Tachograph Speed [km/h]',
                 'IVH - GPS position']

for i in range(len(step_signal)//2):
    
    temp = X.loc[head_list[i]:tail_list[i], final_feature]
    temp.reset_index(drop=True, inplace=True)
    first_nonzero_index = temp['Vehicle - Tachograph Speed [km/h]'].nonzero()[0][0]
    last_nonzero_index = temp['Vehicle - Tachograph Speed [km/h]'].nonzero()[0][-1]
    temp = temp[first_nonzero_index:last_nonzero_index]
    temp.reset_index(drop=True, inplace=True)
    # 添加新的一列 ‘loop’
    temp['loop'] = i+1
    segmentation_list_1.append(temp)
    print('loading the {0} segmentation to the list'.format(i+1))
    

### ******* mixed gaussian model to clustering the average speed 
# 整合所有的数据

#loop_list = ['s'+str(i+1) for i in range(len(segmentation_list_1))]
df_merge =[]
df_merge = df=pd.DataFrame(df_merge)
# 以前试过的一种数据结构，多维index，不同的圈用s1,0;s1,1....s12,1来表示，对后面的画图操作不友好所以暂时不用
#df_merge_1 = pd.concat(segmentation_list_1, keys=loop_list)
df_merge = pd.concat(segmentation_list_1)
df_merge.reset_index(drop=True, inplace=True)
df_X = df_merge.copy(deep=True)


window_size = 300
speed_mean = df_merge['Vehicle - Tachograph Speed [km/h]'].rolling(window_size, min_periods=1).mean()
speed_raw = df_merge['Vehicle - Tachograph Speed [km/h]']
df_X['Vehicle - Tachograph Speed [km/h]'] = df_merge['Vehicle - Tachograph Speed [km/h]'].rolling(window_size, min_periods=1).mean()

""" *********** plot raw data and data after moving average ************ """
f, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(np.arange(0,len(df_merge)), speed_raw, label='raw data', color=next(colorset))
ax[1].plot(np.arange(0,len(df_merge)), speed_mean, label='moving average', color=next(colorset))

speed_raw = speed_raw.round(1)
speed_mean = speed_mean.round(1)
df_X['Vehicle - Tachograph Speed [km/h]'] = df_X['Vehicle - Tachograph Speed [km/h]'].round(1)
plt.figure()
plt.hist(speed_mean, bins = 60)

# 每隔固定的间隔，取一个平均值
speed_mean_downsampling = df_X[::window_size]
speed_mean_downsampling.reset_index(drop=True, inplace=True)
plt.figure()
# 画出速度均值的统计直方图
plt.hist(speed_mean_downsampling['Vehicle - Tachograph Speed [km/h]'], bins = 60)
plt.figure()
# 速度随时间的变化图
plt.plot(np.arange(0,len(speed_mean_downsampling)), speed_mean_downsampling['Vehicle - Tachograph Speed [km/h]'])

## 高斯模型确定几个簇，然后开始预测(face book 的预测方法)

# change pandas series to numpy array
speed_array = speed_mean_downsampling['Vehicle - Tachograph Speed [km/h]'].sort_values().values.reshape(-1,1)
gmm = sklearn.mixture.GaussianMixture(n_components=4, n_init=10).fit(speed_array)
labels = gmm.predict(speed_array)
plt.scatter(np.arange(0,len(speed_array)), speed_array, c= labels)

def gd(x, mu=0, sigma=1):
    """根据公式，由自变量x计算因变量的值
    Argument:
        x: array
            输入数据（自变量）
        mu: float
            均值
        sigma: float
            方差
    """
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu)**2 / (2 * sigma))
    return left * right

plt.figure()
x_axis = np.arange(0,60,0.1).reshape(-1,1)
# 这里可以选择好看的配色方案
plt.hist(speed_mean_downsampling['Vehicle - Tachograph Speed [km/h]'], bins = 60, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.plot(x_axis, 50*gd(x_axis, gmm.means_[0], gmm.covariances_[0]))
plt.plot(x_axis, 160*gd(x_axis, gmm.means_[1], gmm.covariances_[1]))
plt.plot(x_axis, 300*gd(x_axis, gmm.means_[2], gmm.covariances_[2]))
plt.plot(x_axis, 200*gd(x_axis, gmm.means_[3], gmm.covariances_[3]))
plt.ylim((0,50))


# ****** 尝试使用对数化，效果不是很好 ********
#plt.plot(np.arange(0,len(speed_mean_downsampling)), np.log(speed_mean_downsampling+1))
loop_ind_seg = [0]
for i in range(len(segmentation_list_1)):
    
    ind = np.nonzero(speed_mean_downsampling['loop'] == (i+1))[0][-1]
    loop_ind_seg.append(ind+1)

""" 分段画图，展示一天中不同的圈 """

plt.figure('分段画图')
for i in range(len(segmentation_list_1)):
    
    speed_mean_downsampling.loc[loop_ind_seg[i]:loop_ind_seg[i+1],'Vehicle - Tachograph Speed [km/h]'].plot(label='loop '+str(i+1))

plt.legend(loc='best', ncol=6)
plt.xticks([0,20,40,60,80,100,120],['5:15','7:37','10:13','12:52','14:56','17:24','20:00'])










