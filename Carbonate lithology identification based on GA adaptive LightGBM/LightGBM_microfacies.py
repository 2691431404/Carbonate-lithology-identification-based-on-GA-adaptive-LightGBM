import pandas as pd
import numpy as np
import datetime
import time
import math
import datetime
import lightgbm
import random
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

'''
    Import Data
'''
# df = pd.read_csv('test.csv')
df = pd.read_csv('../data/data_well_id.csv')
columns =['Facies','well_id','GR','ILD_log10','DeltaPHI','PHIND','PE']
df = df[columns]
print("Shape df : ", df.shape)
df.head()


lab =df['Facies']
df = df.drop(columns=['Facies'])
# standardization in (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
df = pd.DataFrame(scaler.fit_transform(df),columns=['well_id','GR','ILD_log10','DeltaPHI','PHIND','PE'])
df = pd.concat([df,lab],axis=1)

print(df.head())

'''
交会图
第二句就是绘图啦~kind表示联合分布图中非对角线图的类型，可选'reg'与'scatter'、'kde'、'hist'，
'reg'代表在图片中加入一条拟合直线，'scatter'就是不加入这条直线,'kde'是等高线的形式，
'hist'就是类似于栅格地图的形式；diag_kind表示联合分布图中对角线图的类型，
可选'hist'与'kde'，'hist'代表直方图，'kde'代表直方图曲线化。
  以kind和diag_kind分别选择'reg'和'kde'为例，
'''
def figure(data):

    joint_columns = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE','Facies']
    data = data[joint_columns]
    print(data)
    sns.pairplot(data,kind='reg',diag_kind='kde',hue='Facies')
    sns.set(font_scale=1.2)

figure(df)

'''
     Machine Learning
        1 - Parameters & Evaluation function
'''
param_classifieur = {'boosting_type': 'goss',# 设置提升类型
                     'colsample_bytree': 1,
                     'learning_rate':0.05,# 学习速率
                     'max_depth': 5,
                     'verbose':10,
                     'n_jobs': 4,
                     'num_leaves': 30,# 叶子节点数
                     'objective': 'multiclass',# 目标函数
                     'num_class':9,
                     'random_state': None,
                     'reg_alpha': 0,
                     'reg_lambda': 0,
                     'subsample_for_bin': 1100,
                     'max_bin':255,
                     'metric':'None',
                     'boost_from_average':True, #IMPORTANT !
                     'use_missing':True,
                     'is_unbalance' :True}

def acc_eval(preds, train_data):
    """ Accuracy evaluation function"""
    n_labels = 9
    y_true = train_data.get_label()

    reshaped_preds = preds.reshape(n_labels, -1)
    print(reshaped_preds.shape)
    y_pred = np.argmax(reshaped_preds, axis=0)
    acc = accuracy_score(y_true, y_pred)
    return 'accuracy', acc, True

'''
        2 - Training
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

num_model = 3

print("Start creation of Model number : " + str(num_model))

# Keep index
print("Find index...")
# well_id_train = []
# well_id_validation = []
# well_id_validation = [ind for ind in range(500*(num_model-1), 500*num_model, 1)]
# print(well_id_validation)
# well_id_train = [ind for ind in range(0, 4000, 1) if ind not in well_id_validation]
# print(well_id_train)
# ind_train = df.index[(df['Well Name'] >= min(well_id_train))&(df['Well Name'] <= max(well_id_train))]
# print(ind_train.shape[0])
# ind_validation = df.index[(df['Well Name'] >= min(well_id_validation))&(df['Well Name'] <= max(well_id_validation))]

print("Segmentation...")
# Segmentation
# x_train = pd.DataFrame(df.loc[ind_train, features], columns = features)
# y_train = pd.DataFrame(df.loc[ind_train, target], columns=[target])
# x_validation = pd.DataFrame(df.loc[ind_validation, features], columns = features)
# y_validation = pd.DataFrame(df.loc[ind_validation, target], columns = [target])
Y = df.Facies - 1
data1 = df.drop(columns=['Facies'])
x_train, x_validation, y_train, y_validation = train_test_split(data1, Y, test_size=0.2, random_state=0,
                                                                stratify=Y)  # 随机采样30%的数据样本作为测试集
# #Create LightGBM Dataset
lgb_train = lightgbm.Dataset(x_train.values, y_train.values)
lgb_validation = lightgbm.Dataset(x_validation.values, y_validation.values)
print("Start training model : " + str(num_model) + ' ...')
# Training
evals_results = {}
model = lightgbm.train(params=param_classifieur,
                       train_set=lgb_train,
                       valid_sets=[lgb_validation],
                       valid_names=['Validation'],
                       evals_result=evals_results,
                       feval=acc_eval,
                       num_boost_round=5000,
                       early_stopping_rounds=50)

# print("End training -> save model")
model.save_model('model_' + str(num_model), num_iteration=model.best_iteration)
# 8.使用模型对测试集数据进行预测
predictions = model.predict(x_validation, num_iteration=model.best_iteration)

# 9.对模型的预测结果进行评判（平均绝对误差）
reshaped_preds = predictions.reshape(9, -1)
y_pred = np.argmax(reshaped_preds, axis=0)
print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_validation)))



model = lightgbm.Booster(model_file='model_3')

prediction_train = pd.DataFrame(model.predict(x_train), columns=['Proba_'+str(i) for i in range(0,9,1)],
                                index=x_train.index)
prediction_validation = pd.DataFrame(model.predict(x_validation), columns=['Proba_'+str(i) for i in range(0,9,1)],
                                     index=x_validation.index)

def convertProbaToInteger(prediction):
    return np.argmax(prediction.T.values, axis=0)

int_pred_train = pd.DataFrame(prediction_train.apply(convertProbaToInteger, axis=1), columns=['Prediction'],
                              index=x_train.index)
print(int_pred_train)
int_pred_val = pd.DataFrame(prediction_validation.apply(convertProbaToInteger, axis=1), columns=['Prediction'],
                            index=x_validation.index)
print('\nModel:')
print('train accuracy :', round(accuracy_score(y_true=y_train, y_pred=int_pred_train), 3))
print('validation accuracy :', round(accuracy_score(y_true=y_validation, y_pred=int_pred_val), 3))
print(classification_report(y_validation, int_pred_val))


