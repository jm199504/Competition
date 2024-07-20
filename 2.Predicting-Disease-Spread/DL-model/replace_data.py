# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:22:27 2019

@author: Junming
"""
import pandas as pd
import numpy as np

#将NaN填补为0以及保留至少15个特征的数据
def replace_zero(data_path,dest_path,thresh):
    data = pd.read_csv(data_path)
    if 'total_case' in data.columns:
        data = data.dropna(thresh=thresh)
    data = data.replace(to_replace=np.nan,value=0)
    data.to_csv(dest_path,index=None)
    print('填补NaN为0值:finished')

#缺失值处理train:选择前
def train_rn_krd(data_path,dest_path,krd):
    data = pd.read_csv(data_path)
    columns = list(data.columns)
    columns.remove('total_cases')
    for column in columns:
        for demo in range(len(data[column])):
            if data[column][demo] == 0:
                temp_list = list()
                if krd<demo<len(data[column])-krd:
                    for i in np.arange(demo - krd, demo):
                        if data[column][i] > 0:
                            temp_list.append(data[column][i])
                    if len(temp_list) == 0:
                        data[column][demo] = np.mean(data[column])+ np.random.choice\
                        (np.arange(-np.std(data[column],ddof=1), np.std(data[column],ddof=1), 0.01))
                    else:
                        temp_mean = np.mean(temp_list)
                        temp_std = np.std(temp_list, ddof=1)
                        temp_std = 0.01 if np.isnan(temp_std) or temp_std == 0 else temp_std
                        data[column][demo] = temp_mean + np.random.choice(np.arange(-temp_std, temp_std, 0.01))
                else:
                    data[column][demo] = 0
#    data = data.drop(columns=['city','year','weekofyear','week_start_date','precipitation_amt_mm'])
    data.to_csv(dest_path,index=None)
    print('使用前后N周数据±标准差随机填补0值:finished')
    
def train_rn_klabel(data_path,dest_path):
    data = pd.read_csv(data_path)
    remove_total_columns = list(data.columns)
    remove_total_columns.remove('total_cases')
    for column in remove_total_columns:
        for i in range(len(data[column])):
            if data[column][i]==0:
                feature_list = list()
                label_value = data['total_cases'][i]
                for z in range(1, 6):
                    label_list = list(np.arange(label_value - z, label_value + z))
                    for m in range(len(data['total_cases'])):
                        if data['total_cases'][m] in label_list:
                            feature_list.append(data[column][m])
                    if len(feature_list)>0:
                        break
                if len(feature_list)==0:# 将此标签置为异常数据样本
                    data['total_case'][i] = -1
                else:
                    data[column][i] = np.mean(feature_list) + np.random.choice(np.arange(-np.std(feature_list,ddof=1), np.std(feature_list,ddof=1), 0.01))
    data = data.drop(columns=['year','weekofyear','week_start_date','precipitation_amt_mm'])
    data.to_csv(dest_path,index=None)
    print('使用相同标签±标准差随机填补0值:finished')
    
def train_fillna(data_path,destination_path):
    data = pd.read_csv(data_path)
    data.fillna(method='ffill',inplace=True)
#    data = data.drop(columns=['year', 'weekofyear', 'week_start_date', 'precipitation_amt_mm'])
    data.to_csv(destination_path,index=None)
    print('使用上一周非NaN:finished')

def test_rn_krd(data_path,dest_path):
    data = pd.read_csv(data_path)
    for column in data.columns:
        for i in range(len(data[column])):
            if data[column][i] == 0:
                dc_list = list()
                for m in range(len(data[column])):
                    if data[column][m]>0:
                        dc_list.append(data[column][m])
                data[column][i] = np.mean(dc_list) + np.random.choice(np.arange(-np.std(dc_list,ddof=1), np.std(dc_list,ddof=1), 0.01))
    data.to_csv(dest_path, index=None)
    print('测试集使用前后N周数据±标准差随机填补0值:finished')
    