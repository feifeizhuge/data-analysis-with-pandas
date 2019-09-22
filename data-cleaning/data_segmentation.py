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
import sklearn.mixture
import math
import itertools
import matplotlib.ticker as ticker


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
       'Vehicle - High resolution vehicle distance [km]'
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
        ax[ind].legend(loc='upper right')
#        ax[1].set_xlabel('Data points', fontsize=14)
#        ax[ind].set_ylabel(feature_list[ind], fontsize=14)
        
    
plot_feature(X, battery)
plot_feature(X, primove)
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
X['Primove - Pick-up position control'] = signal.medfilt(X['Primove - Pick-up position control'], 11)
X.reset_index(drop=True, inplace=True)

""" *********** 分段化圈原理展示图 ************ """
plt.figure('分段划圈展示图')
plt.plot(X['Vehicle - Tachograph Speed [km/h]'])
plt.plot(X['Primove - Pick-up position control'] * 30)
plt.plot(X['Primove - Pick-up position control'].rolling(2,min_periods = 1).mean() * 70,'.')
plt.legend(['Tachograph Speed [km/h]','Primove - Pick-up position control','Mean of Pick-up position control'],loc='upper right')
plt.xlabel('Data points', fontsize=12)
plt.ylabel('Tachograph Speed [km/h] & Pick-up Position Control Signal[1]',fontsize=12)

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

loop_list = ['s'+str(i+1) for i in range(len(segmentation_list_1))]
df_merge =[]
df_merge = df=pd.DataFrame(df_merge)
# 以前试过的一种数据结构，多维index，不同的圈用s1,0;s1,1....s12,1来表示，对后面的画图操作不友好所以暂时不用
df_merge_1 = pd.concat(segmentation_list_1, keys=loop_list)
df_merge = pd.concat(segmentation_list_1)
df_merge.reset_index(drop=True, inplace=True)
df_X = df_merge.copy(deep=True)


window_size = 300
speed_mean = df_merge['Vehicle - Tachograph Speed [km/h]'].rolling(window_size, min_periods=1).mean()
speed_raw = df_merge['Vehicle - Tachograph Speed [km/h]']
df_X['Vehicle - Tachograph Speed [km/h]'] = df_merge['Vehicle - Tachograph Speed [km/h]'].rolling(window_size, min_periods=1).mean()

""" *********** plot raw data and data after moving average ************ """
f, ax = plt.subplots(2,1,sharex=True,num='原始速度曲线 & 滑动平均速度曲线')
ax[0].plot(np.arange(0,len(df_merge)), speed_raw, label='raw data', color="#1f77b4")
ax[0].legend(loc='upper right')
ax[0].set_ylabel('Tachograph Speed [km/h]', fontsize=12)
ax[1].plot(np.arange(0,len(df_merge)), speed_mean, label='moving average', color="#ff7f0e")
ax[1].legend(loc='upper right')
ax[1].set_xlabel('Data points', fontsize=12)
ax[1].set_ylabel('Moving Average speed in priod of time/'+ str(int(window_size/60)) +' minutes', fontsize=12)

speed_raw = speed_raw.round(2)
speed_mean = speed_mean.round(2)
df_X['Vehicle - Tachograph Speed [km/h]'] = df_X['Vehicle - Tachograph Speed [km/h]'].round(2)
plt.figure()
plt.hist(speed_mean, bins = 60)

# 每隔固定的间隔，取一个平均值
speed_mean_downsampling = df_X[::window_size]
speed_mean_downsampling.reset_index(drop=True, inplace=True)
plt.figure('速度均值的统计直方图')
# 画出速度均值的统计直方图
plt.hist(speed_mean_downsampling['Vehicle - Tachograph Speed [km/h]'], bins = 30)
plt.figure('速度随时间的变化图')
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

plt.figure('速度分布统计直方图，和混合高斯分布')
x_axis = np.arange(0,60,0.1).reshape(-1,1)
# 这里可以选择好看的配色方案
# *********** 速度聚类结果---展示图 ******************
plt.hist(speed_mean_downsampling['Vehicle - Tachograph Speed [km/h]'], bins = 30, density=0, facecolor="#1f77b4",\
         edgecolor="black", alpha=0.7)
plt.plot(x_axis, 20*gd(x_axis, gmm.means_[3], gmm.covariances_[3]), label='Cluster 1')
plt.plot(x_axis, 130*gd(x_axis, gmm.means_[1], gmm.covariances_[1]), label='Cluster 2')
plt.plot(x_axis, 100*gd(x_axis, gmm.means_[2], gmm.covariances_[2]), label='Cluste 3')
plt.plot(x_axis, 50*gd(x_axis, gmm.means_[0], gmm.covariances_[0]), label='Cluster 4')
plt.xlim([0,55])
plt.ylim([0,40])
plt.xlabel('Time', fontsize=14)
plt.ylabel('Average speed in priod of time/'+ str(int(window_size/60)) +' minutes', fontsize=14)
plt.legend()


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
plt.xticks([0,20,40,60,80,100,120],['5:15','7:42','10:18','12:52','15:18','17:29','20:04'])
plt.vlines(x=(0,20,40,60,80,100,120), ymin=-2.0, ymax=48,linestyles='dashed',alpha=0.7,lw=0.5)
plt.ylim([-0.5,47])
plt.xlabel('Time', fontsize=14)
plt.ylabel('Average speed in priod of time/'+ str(int(window_size/60)) +' minutes', fontsize=14)

""" 根据论文 speed index 对交通状况进行量化 """

## ********* 1. 速度的统计条形图 *********** 
# 公交车的最高时速 65[km/h]
max_speed = 60
speed_raw = np.clip(speed_raw,0,60)
# 四舍五入取整数，再转化为array; speed performance index的定义在这里
speed_index_raw = round((speed_raw/max_speed)*100,2).values
bar_parameters = np.histogram(speed_index_raw, bins=20, range=(0,100))
# 累加概率分布图
cpd_parameters = np.cumsum(np.round(100*bar_parameters[0] / sum(bar_parameters[0]), 2))
fig, ax1 = plt.subplots(num='Speed Performance Index 条形图')
ax1.bar(bar_parameters[1][1::], bar_parameters[0], width=2, alpha=0.8, color="#bcbd22")
ax1.set_xlabel('The Speed Performance Index')
ax1.set_ylabel('Frequency')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.set_xlim([-1,103])
ax1.spines['top'].set_visible(False)

# 绘制双y坐标图
ax2 = ax1.twinx()
ax2.plot(bar_parameters[1][1::], cpd_parameters,marker='D',alpha=0.8)
ax2.set_ylabel('Cumulative Probiblity Density(%)')
ax2.spines['top'].set_visible(False)

""" 根据speed performance index的定义，定义Road segment congestion index """

speed_index_raw = np.c_[speed_index_raw, df_merge.iloc[:,4].values]
road_segment_cong_ind = []
loop_num = len(segmentation_list_1)
for i in range(loop_num):
    
    loop = speed_index_raw[speed_index_raw[:,1] == (i+1),0]
    R_c = (sum(loop<=75) / len(loop)) * np.mean(loop) / 100
    road_segment_cong_ind.append(R_c)
    print('Road Segment Congestion Index for loop ' + str(i+1) + 'calculated!')

# "coolwarm","RdBu_r"
plt.figure('分圈速度条形折线图')
pal = sns.color_palette("RdBu_r", len(road_segment_cong_ind))
rank = np.asarray(road_segment_cong_ind).argsort().argsort()  
sns.barplot(x=np.arange(1,25), y=road_segment_cong_ind, palette=np.array(pal[::-1])[rank])
plt.plot(np.arange(0,24), road_segment_cong_ind,'.',linestyle='-',alpha=0.6,color='black')
plt.xlabel('Bus Loop', fontsize=14)
plt.ylabel('Road Segment Congestion Index', fontsize=14)

""" 根据speed performance index的定义，定义每个小时，时间的segment congestion index """

    






