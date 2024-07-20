# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:35:56 2019

@author: Junming
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 分析 特征及标签间的相关系数矩阵
def correclation_matrix(data_path,dest_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=['city','year','weekofyear','week_start_date'])
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features',y=1.05,size=15)
    sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,linecolor='white',annot=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=360)
    plt.show()
    corr_result = pd.DataFrame.corr(data,method='pearson')
    corr_result.to_csv(dest_path)
# 分析 可视化各月标签范围
def month_average(data_path):
    data = pd.read_csv(data_path)
    cases_dict = dict()
    for i in range(len(data['week_start_date'])):
        ckey = str(data['week_start_date'][i]).split('/')[1]
        if ckey in cases_dict.keys():
            cases_dict[ckey].append(data['total_cases'][i])
        else:
            cases_dict[ckey] = [data['total_cases'][i]]
    for m in cases_dict.keys():
        plt.bar(np.arange(len(cases_dict[m])),cases_dict[m])
        plt.title('The '+str(m)+' month')
        plt.show()
    # plt.bar(data['week_start_date'],data['total_cases'])
    # plt.show()
# 分析 特征方差值
def features_means(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=['city','year','weekofyear','week_start_date'])
    for column in data.columns:
        print(column+'\t'+str(np.std(data[column],ddof=1)))
        
# 等宽法分析
def total_cases_distribution(data_path):
    data = pd.read_csv(data_path)
    iq_total_cases_dict= dict()
    sj_total_cases_dict= dict()
    for i in range(len(data['city'])):
        if data['city'][i] == 'iq':
            if data['total_cases'][i] in iq_total_cases_dict.keys():
                iq_total_cases_dict[data['total_cases'][i]]+=1
            else:
                iq_total_cases_dict[data['total_cases'][i]]=1
        if data['city'][i] == 'sj':
            if data['total_cases'][i] in iq_total_cases_dict.keys():
                sj_total_cases_dict[data['total_cases'][i]]+=1
            else:
                sj_total_cases_dict[data['total_cases'][i]]=1
    print(iq_total_cases_dict)
    print('='*30)
    print(sj_total_cases_dict)
    for k,v in iq_total_cases_dict.items():
        plt.scatter(k,v)
    plt.show()
    
    
