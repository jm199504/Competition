## 驾驶行为预测驾驶风险

### 1.竞赛链接

https://www.datafountain.cn/competitions/284

### 2.竞赛排名 

<img src="https://github.com/jm199504/Competition/blob/master/Driving%20behavior%20predicts%20driving%20risk/images/1.png">

（截止于2018.07.10）

### 3.数据介绍

<img src="https://github.com/jm199504/Competition/blob/master/Driving%20behavior%20predicts%20driving%20risk/images/2.png">

### 4.数据观察

#### 4.1地理位置特征

根据多条用户行程目前所在经度/纬度变化，计算行驶距离，计算地球的两点（已知经纬度）实际距离，计算公式如下：

<img src="https://github.com/jm199504/Competition/blob/master/Driving%20behavior%20predicts%20driving%20risk/images/3.png" width="500">

实现代码如下：

```
def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000
```

#### 4.2记录条数特征

每个用户存在多条行驶记录（并不代表多次行驶记录，因为一次行驶记录可能包含多条行驶记录，数据提供是时间点数据），因此用户行驶次数同样可视为其数据特征：

```
# trip 特征
num_of_trips = temp['TRIP_ID'].nunique()  # Return number of unique elements in the object
```

#### 4.3用户通话特征

数据特征中包含用户行程目前的通话状态。（0,未知 1,呼出 2,呼入 3,连通 4,断连），统计用户各通话状态的频率：

```
# record 特征
num_of_records = temp.shape[0]
num_of_state = temp[['TERMINALNO', 'CALLSTATE']]
nsh = num_of_state.shape[0]
num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE'] == 0].shape[0] / float(nsh)
num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE'] == 1].shape[0] / float(nsh)
num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE'] == 2].shape[0] / float(nsh)
num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE'] == 3].shape[0] / float(nsh)
num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE'] == 4].shape[0] / float(nsh)
```

#### 4.4用户驾驶时长特征

提供了行驶记录的unix时间戳（从1970年1月1日（UTC/GMT的午夜）开始所经过的秒数，不考虑闰秒），因此可以考虑统计用户的行驶时长，首先需要将时间戳转为日期格式：

```
temp['hour'] = temp['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
```

#### 4.5驾驶区间特征

夜晚行驶危险性会有所提升，因而我们考虑将晚上20点——凌晨5点作为夜晚行驶记录（特征衍生）

#### 4.6驾驶速度

计算一次驾驶行为（含该次一条及以上记录）的行驶速度方差和平均速度

### 5.模型考虑

尝试了LGBMRegressor、XGBRegressor、RandomForestRegressor



总结：该比赛是研一期间和实验室小伙伴一起参与，没有经验丰富的老手带领，所以在尝试摸索和模型参数手动优化下拿到了75/2749，个人认为首次参赛已经很满足，其实可以还可以考虑模型融合，另外未筛选验证集使得调参均为人工尝试，很怀念当时实验室同小伙伴一起讨论和寻找新的特征场景，马上进入秋招和毕业年，希望小伙伴们拿到满意的offer和顺利通过毕业答辩~耶！
