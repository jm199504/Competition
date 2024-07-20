# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:20:46 2019

@author: Junming
"""
import pandas as pd

#合并标签
def merge(x_path,y_path,dest_path):
    features = pd.read_csv(x_path)
    labels = pd.read_csv(y_path)
    features = features.merge(labels,how='inner')
    features = features.drop(columns=['precipitation_amt_mm'])
    features.to_csv(dest_path,index=None)
    print('合并标签:finished')
#根据城市划分数据训练集/测试集
def split_cities(data_path,iq_path,sj_path):
    train_merge = pd.read_csv(data_path)
    iq_train,sj_train = train_merge[train_merge.city == 'iq'],train_merge[train_merge.city == 'sj']
    iq_train.to_csv(iq_path,index=None)
    sj_train.to_csv(sj_path,index=None)
    print('根据城市划分数据训练集/测试集:finished')