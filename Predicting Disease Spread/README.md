**Predicting Disease Spread**

**竞赛链接**

<https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/>

**竞赛排名**（截止于2019.06.06）GRU模型

<img src="https://github.com/jm199504/Competition/blob/master/Predicting%20Disease%20Spread/images/rank.png" width="500">

**数据介绍**

<img src="https://github.com/jm199504/Competition/blob/master/Predicting%20Disease%20Spread/images/features.png" width="600">

**目录**

1. DL-model：使用GRU模型（含LSTM和SimpleRNN-注释代码）
   1. py文件
   2. 运行main.py（含运行流程）
   3. 2个城市分别训练（iq_gru.py&sj_gru.py）

2. ML-model：使用LGBM模型（含RF、SVR、GBR、XGBR、CatBR-注释代码）
   1. ipynb文件
   2. _Add_Features文件表添加衍生特征

**处理流程**

**数据集缺失值处理办法**

1. 样本缺失特征数量超过一定阈值删除
2. 选用fillna（method='ffill',inplace=True）
3. 选用该列均值±1×标准差   [具有随机性，模型结果不稳定]
4. 使用列均值/中位数/众数填补 [均值填补会受到存在异常值的影响]
5. 选用total_cases相同的均值，无相同则考虑±(1~n)寻找均值  
6. 线性插值法最邻近插值法/阶梯插值/2、3阶B样条曲线插值   
7. KNN填充 
8. 随机森林填充   

ML-model-代码展示方法3

<img src="https://github.com/jm199504/Competition/blob/master/Predicting%20Disease%20Spread/images/fill.png">

**添加衍生数据特征**

1. 由于该赛题是时间序列数据（以week为最小时间单位），将前n周的特征复制到当周的特征中
2. 计算特征与均值/中位数的差值 
3. 计算最高温度与最低温度的差值；当周的植被总和；

ML-model-代码展示方法2

<img src="https://github.com/jm199504/Competition/blob/master/Predicting%20Disease%20Spread/images/add.png">

**数据归一化**

1. 最大最小归一化
2. 标准归一化

**数据平滑**

1. 个别样本total_cases相比其他样本偏离过大，远超于均值±3倍标准差范围，因而考虑删除部分样本

**数据特征排名**

1. 随机森林进行特征排名

<img src="https://github.com/jm199504/Competition/blob/master/Predicting%20Disease%20Spread/images/feature_rank.png">

**模型选择**

1. GradientBoostingRegressor
2. RandomForestRegressor
3. LGBMRegressor
4. Catboost
5. SVR
6. SimpleRNN
7. LSTM
8. GRU

**参数寻优**

1. 选择GridSearchCV进行模型网格最优参数搜索

**预测结果处理**

1. 由于大量total_cases为0，模型预测值可能出现负数，对其置为0
2. 不处理none/向下取整floor/向上取整ceil/四舍五入round

**总结**

1. 缺失值填补：选用该列均值±1×标准差进行缺失值填补
2. 数据平滑：远超于均值±5倍模型效果有所提升
3. DL最佳模型：GRU
4. ML最佳模型：LGBMRegressor
5. 其他：Catboost与LGBM预测效果相近，但计算时间远大于LGBM
6. 数据处理：向下取整模型效果有所提升

**未来工作**

1. 使用随机森林模型/KNN填补缺失值
2. 进行更多特征衍生
3. 尝试其他神经网络模型
