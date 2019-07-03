import warnings
warnings.filterwarnings("ignore")
from merge_splite import *
from replace_data import *
from utils import *
from iq_nn import *
from sj_nn import *
from analysis import *
from iq_lgb import *
from sj_lgb import *


train_path = 'data\\pre\\train_data\\'
test_path = 'data\\pre\\test_data\\'
result_path = 'data\\result\\'
paths = ['iq','sj']

# 考虑填补周数使用1或3以及均值 //不加入随机///标准归一化
if __name__=='__main__':
     # 合并标签
    merge(x_path='data\\dengue_features_train.csv',y_path='data\\dengue_labels_train.csv',dest_path='data\\pre\\dengue_merge_train.csv')
     # 划分数据
    split_cities(data_path='data\\pre\\dengue_merge_train.csv',iq_path= train_path+'iq_train.csv',sj_path= train_path+'sj_train.csv')
    split_cities(data_path='data\\dengue_features_test.csv',iq_path= test_path+'iq_test.csv', sj_path= test_path+'sj_test.csv') 
    for path in ['iq','sj']:
        # 训练集-缺失值处理
        replace_zero(train_path+path+'_train.csv',train_path+path+'_train_p0.csv',15)
        train_rn_krd(train_path+path+'_train_p0.csv',train_path+path+'_train_p1.csv',krd=2)  
#        train_fillna(train_path+path+'_train.csv',train_path+path+'_train_p1.csv')
     
        #测试集-缺失值处理
        replace_zero(test_path+path+'_test.csv',test_path+path+'_test_p0.csv',15)
        test_rn_krd(test_path+path+'_test_p0.csv',test_path+path+'_test_p1.csv')
        
        # 特征添加
        add_features(train_path + path + '_train_p1.csv',train_path + path + '_train_p2.csv',3)
        add_features(test_path + path + '_test_p1.csv',test_path + path + '_test_p2.csv',3)

    # 预测模型
#    iq_lgb_model(train_path+'iq_train_p2.csv',test_path+'iq_test_p2.csv',result_path+'iq_result.csv')
#    sj_lgb_model(train_path+'sj_train_p2.csv',test_path+'sj_test_p2.csv',result_path+'sj_result.csv')
    iq_nn_model(train_path+'iq_train_p2.csv',test_path+'iq_test_p2.csv',result_path+'iq_result.csv')
    sj_nn_model(train_path+'sj_train_p2.csv',test_path+'sj_test_p2.csv',result_path+'sj_result.csv')
    
    # 合并测试集结果
    merge_result(result_path+'sj_result.csv',result_path+'iq_result.csv',result_path+'result.csv')
    # 删除异常数据
    delete_exresult(result_path +'result.csv')         
    