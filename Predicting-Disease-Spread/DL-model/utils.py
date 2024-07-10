# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:38:40 2019

@author: Junming
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np

# 训练集特征选择
def features_choice(data_path,num):
    data = pd.read_csv(data_path)
    clf = RandomForestRegressor(oob_score=True, n_estimators=1000)
    features_list = list(data.columns)
    for f in ['city','year','weekofyear','week_start_date','total_cases']:
        features_list.remove(f)
    clf = clf.fit(data[features_list],data['total_cases'])
    importances = clf.feature_importances_
    features_dict = dict()
    for i,j in zip(features_list,importances):
        features_dict[i] = j
    fsort = sorted(features_dict.items(),key=lambda item:item[1])
    remove_list = list()
    for f in fsort[:num]:
        remove_list.append(f[0])
    return remove_list
    # plt.bar(features_list,importances)
    # plt.title(data_path[-12:-10])
    # plt.show()
    print('特征选择:finished')
    
#def PCA_choosefeature(data_path,dest_path,nums):
#    data = pd.read_csv(data_path)
#    estimator = PCA(n_components=nums)
#    for f in ['city','year','weekofyear','week_start_date','total_cases']:
#        data = data.drop(columns=f)
#    X = data.values
#    pca_X = estimator.fit_transform(X)
    
    
# 删除训练集和测试集对应特征
def delete_features(data_path,remove_list):
    data = pd.read_csv(data_path)
    data = data.drop(columns = remove_list)
#    print(data.columns)
    data.to_csv(data_path,index=None)
# 添加特征
def add_features(data_path,dest_path,week):
    data = pd.read_csv(data_path)
    features = list(data.columns)
    remove_list = list()
    for i in ['city','year','weekofyear','week_start_date','total_cases']:
        if i in features:
            remove_list.append(i)
    for j in remove_list:
        features.remove(j)
        
    for w in range(week):
        for f in features:
            data[str(w+1)+'-'+f] = data.groupby('city')[f].shift(w+1)
    for i in range(week):
        for f in data.columns:
            if pd.isnull(data[f][i]):
                if not pd.isnull(data['1'+f[1:]][i]):
                    data[f][i] = data['1'+f[1:]][i]
                else:
                   data[f][i] = data[f[2:]][i]
    # 添加新的特征
    air_temp_diff1 = list()
    air_temp_diff2 = list()
    total_ndvi = list()
    air_temp_avg1 = list()
    air_temp_avg2 = list()
    air_temp_avg3 = list()
    precip_avg1 = list()
    precip_avg2 = list()
    precip_avg3 = list()
    humidity_avg1 = list()
    humidity_avg2 = list()
    point_avg = list()
    tdtr_avg = list()
    diur_avg = list()
    
    air_temp_avg_value1 = np.mean(data['reanalysis_avg_temp_k'])
    air_temp_avg_value2 = np.mean(data['station_avg_temp_c'])
    air_temp_avg_value3 = np.mean(data['reanalysis_air_temp_k'])
    
    precip_avg_value1 = np.mean(data['reanalysis_sat_precip_amt_mm'])
    precip_avg_value2 = np.mean(data['reanalysis_precip_amt_kg_per_m2'])
    precip_avg_value3 = np.mean(data['station_precip_mm'])
    
    humidity_avg_value1 = np.mean(data['reanalysis_relative_humidity_percent'])
    humidity_avg_value2 = np.mean(data['reanalysis_specific_humidity_g_per_kg'])
    
    point_avg_value = np.mean(data['reanalysis_dew_point_temp_k'])
    
    tdtr_avg_value = np.mean(data['reanalysis_tdtr_k'])
    diur_avg_value = np.mean(data['station_diur_temp_rng_c'])
    
    for i in range(len(data['city'])):
        # (温差1)
        air_temp_diff1.append(data['reanalysis_max_air_temp_k'][i]-data['reanalysis_min_air_temp_k'][i])
        # (温差2)
        air_temp_diff2.append(data['station_max_temp_c'][i]-data['station_min_temp_c'][i])
        # (植被总和)
        temp_sum = data['ndvi_se'][i]+data['ndvi_sw'][i]+data['ndvi_ne'][i]+data['ndvi_nw'][i]
        total_ndvi.append(temp_sum)
        # (与均温差)
        air_temp_avg1.append(data['reanalysis_avg_temp_k'][i]-air_temp_avg_value1)
        air_temp_avg2.append(data['station_avg_temp_c'][i]-air_temp_avg_value2)
        air_temp_avg3.append(data['reanalysis_air_temp_k'][i]-air_temp_avg_value3)
        # (与均降水差)
        precip_avg1.append(data['reanalysis_sat_precip_amt_mm'][i]-precip_avg_value1)
        precip_avg2.append(data['reanalysis_precip_amt_kg_per_m2'][i]-precip_avg_value2)
        precip_avg3.append(data['station_precip_mm'][i]-precip_avg_value3)
        # (与均湿度差)
        humidity_avg1.append(data['reanalysis_relative_humidity_percent'][i]-humidity_avg_value1)
        humidity_avg2.append(data['reanalysis_specific_humidity_g_per_kg'][i]-humidity_avg_value2)
        # (与均露点差)
        point_avg.append(data['reanalysis_dew_point_temp_k'][i]-point_avg_value)
        # (与均温差日较差)tdtr
        tdtr_avg.append(data['reanalysis_tdtr_k'][i]-tdtr_avg_value)
        # (与均温差日较差)diur
        diur_avg.append(data['station_diur_temp_rng_c'][i]-diur_avg_value)
        
    data['air_temp_diff1'] = air_temp_diff1
    data['air_temp_diff2'] = air_temp_diff2
    data['total_ndvi'] = total_ndvi
    data['air_temp_avg1'] = air_temp_avg1 
    data['air_temp_avg2'] = air_temp_avg2
    data['air_temp_avg3'] = air_temp_avg3 
    data['precip_avg1'] = precip_avg1 
    data['precip_avg2'] = precip_avg2
    data['precip_avg3'] = precip_avg3
    data['humidity_avg1'] = humidity_avg1 
    data['humidity_avg2'] = humidity_avg2 
    data['point_avg'] = point_avg 
    data['tdtr_avg'] = tdtr_avg 
    data['diur_avg'] = diur_avg 
            
    data.to_csv(dest_path,index=None)
    print('特征添加:finished')


# 合并测试集结果
def merge_result(sj_path,iq_path,result_path):
    sj_result = pd.read_csv(sj_path)
    iq_result = pd.read_csv(iq_path)
    result = sj_result.append(iq_result)
    final_features_list = ['city','year','weekofyear','total_cases']
    del_features_list = list(set(result.columns).difference(set(final_features_list)))
    result = result.drop(columns=del_features_list)
    result = result[final_features_list]
    
    result.to_csv(result_path,index=None)
    print('合并测试集结果:finished')
    
# 删除异常结果集
def delete_exresult(result_path):
    result = pd.read_csv(result_path)
   
    for i in range(len(result['total_cases'])):
        if result['total_cases'][i] < 0:
            result['total_cases'][i] = 0
    print("结果已生成!!!")
#    print(result)
    result.to_csv(result_path,index=None)
    
 # 删除异常训练数据集
#def delete_exresult(result_path):
#    result = pd.read_csv(result_path)
#    # 惩罚iq
#    for i in range(len(result['total_cases'])):
#        if result['city'][i] == 'iq':
#            result['total_cases'][i]-=2
#    for i in range(len(result['total_cases'])):
#        if result['total_cases'][i] < 0:
#            result['total_cases'][i] = 0
#    
#    print(result)
#    result.to_csv(result_path,index=None)   
 # 删除最大total_cases的样本
#        maxindex = np.argmax(list(train_data['total_cases']))
#        print(maxindex)
#        print(train_data['total_cases'][maxindex])
#        train_data = train_data.drop(index=[maxindex])
#        maxindex = np.argmax(list(train_data['total_cases']))
#        print(maxindex)
#        print(train_data['total_cases'][maxindex])
#        print(type(train_data['total_cases'][maxindex]))
        # ==========
    