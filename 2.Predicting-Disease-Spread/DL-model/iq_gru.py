# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:18:24 2019

@author: Junming Guo

Email: 2017223045154@stu.scu.edu.cn

Location: Chengdu, 610065 Sichuan Province, P. R. China
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,SimpleRNN,GRU
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

np.random.seed(1113)

weeks = 10   # 训练周数
activation='hard_sigmoid' #激活函数hard_sigmoid
optimizer = 'rmsprop' #优化器['rmsprop','adam','nadam','adadelta']#adam
units = 20 #神经元个数
epochs = 50 #迭代次数
validation_split = 0.2 #验证集大小
batch_size = 30 #批次大小
verbose = 0 #不显示训练过程
pshow = True #是否可视化

# iq预测模型
def iq_nn_model(train_path,test_path,result_path):  
    # 读取数据预处理
    train_data,test_data = pd.read_csv(train_path),pd.read_csv(test_path)
    # 删除最大值
#    n = 1
#    for i in range(n):
#        train_data = train_data.drop(index=[np.argmax(train_data['total_cases'])])
    flist = list(train_data.columns)
    train_data.reset_index(inplace=True)
    raw_test_data = test_data.copy()
    remove_list = ['city','year','weekofyear','total_cases','week_start_date']
    for r in remove_list:
        flist.remove(r)
    train_day = int((len(train_data['total_cases']) -weeks + 1))
    test_day = int((len(test_data[flist[0]]) + 1))
    # 将训练集后weeks数据添加到测试集
    temp_datas = train_data.tail(weeks)
    test_data = pd.concat([temp_datas, test_data])
    test_data.reset_index(drop=True, inplace=True)
    flist.append('total_cases')    # 添加total_cases特征
    X_data, Y_data, X_test = list(), list(), list()
    # 训练集：数据处理（所有特征归一化+step分割特征）
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    for property in flist:
        train_data[property] = scaler1.fit_transform(np.reshape(np.array(train_data[property]), (-1, 1)))
    flist.remove('total_cases')    # 特征剔除标签
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    
    for property in flist:
        test_data[property] = scaler2.fit_transform(np.reshape(np.array(test_data[property]), (-1, 1)))
    for i in range(train_day):
        Y_data.append(train_data['total_cases'][i + weeks - 1])
        for j in range(weeks):
            for m in flist:
                X_data.append(train_data[m][i + j])
    X_data = np.reshape(np.array(X_data), (int(len(X_data)/weeks/len(flist)),weeks,len(flist)))
    for i in range(test_day):
        for j in range(weeks):
            for m in flist:
                X_test.append(test_data[m][i + j])
    X_test = np.reshape(np.array(X_test), (int(len(X_test) / weeks / len(flist)), weeks, len(flist)))
    # LSTM模型
#        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model = Sequential()
#        model.add(LSTM(units=units,input_shape=(X_data.shape[1], X_data.shape[2]),activation=activation))
#        model.add(SimpleRNN(units=units,input_shape=(X_data.shape[1], X_data.shape[2]),activation=activation))
    model.add(GRU(units=units,input_shape=(X_data.shape[1], X_data.shape[2]),activation=activation))
    
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    history = model.fit(X_data, Y_data,validation_split=validation_split,epochs=epochs, \
                        batch_size=batch_size, verbose=verbose)
    # 训练集反归一化
    real_train_data = scaler1.inverse_transform(np.reshape(Y_data, (-1, 1)))
    trainPredict = model.predict(X_data)
    trainPredict = scaler1.inverse_transform(trainPredict)
    lstm_train_prediction = list()
    for i in trainPredict.tolist():
        lstm_train_prediction.append(round(i[0]))
    # 测试集反归一化
    testPredict = model.predict(X_test)
    testPredict = scaler1.inverse_transform(testPredict)
    lstm_test_prediction = list()
    for i in testPredict.tolist():
        lstm_test_prediction.append(round(i[0]))
    # 可视化
    if pshow:    
        # 训练集
        plt.figure(figsize=(10, 8))
        plt.plot(list(real_train_data), color='red', label='Real')
        plt.plot(list(lstm_train_prediction), color='blue', label='lstm')
        plt.legend(loc='upper left')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(str(train_path.split('_')[1][-2:])+' Train data Prediction',fontsize=20)
        plt.show()
        # 测试集
        plt.figure(figsize=(10, 8))
        plt.plot(lstm_test_prediction[:-1],color='red',label='lstm')
        plt.legend(loc='upper left')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(str(train_path.split('_')[1][-2:])+' Test data Prediction')
        plt.show()
        # 验证集
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('Loss',fontsize=16)
        plt.xlabel('Epoch',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(str(train_path.split('_')[1][-2:])+' Loss curve of validation',fontsize=20)
        plt.legend(['training', 'validation'], loc='upper right')
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=16)
        plt.show()

    raw_test_data['total_cases'] = list(lstm_test_prediction[:-1])
    raw_test_data.to_csv(result_path, index=None)