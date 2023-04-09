import numpy as np
from sklearn.model_selection import cross_val_score
from deap import algorithms, base, creator, tools
# 导入数据分割， 模型验证，cv参数搜索，以及lightgbm包
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve, f1_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import lightgbm as lgb
# from genetic import GeneticSearchCV
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from geneticpy import GeneticSearchCV, ChoiceDistribution, LogNormalDistribution, UniformDistribution

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

'''
    I - Import Data
'''
# df = pd.read_csv('test.csv')
df = pd.read_csv('../data/data_well_id.csv')
df = df[['row_id','Facies','well_id','GR','ILD_log10','DeltaPHI','PHIND','PE']]
print("Shape df : ", df.shape)
df.head()


'''
    II - Data Preprocessing
        1 - Add Shift features
'''
nb_shift = 51

def addPreviousShift(data, feat_shifted, n_shift, step):
    for k in range(1, n_shift, step):
        data['shift_'+str(k)] = data[feat_shifted].shift(k)
    return

def addFuturShift(data, feat_shifted, n_shift, step):
    for k in range(1, n_shift, step):
        data['shift_'+str(-k)] = data[feat_shifted].shift(-k)
    return

addPreviousShift(data=df, feat_shifted='GR', n_shift=nb_shift, step=1)
addFuturShift(data=df, feat_shifted='GR', n_shift=nb_shift, step=1)


'''
        2 - NaN & Standardization
'''
def dropNaN(data, n_shift):
    data = data.drop([elem for elem in range(0,nb_shift,1)])
    data = data.drop([elem for elem in data.tail(nb_shift).index])
    data = data.reset_index(drop=True)
    return data

df = dropNaN(data=df, n_shift=nb_shift)

features = [col for col in df.columns if col not in ['row_id', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'Facies']]
target = 'Facies'

# standardization in (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
df[features] = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

print(df.head())
Y=df.Facies-1
data1=df.drop(columns=['Facies'])

''' 数据分割'''
x_train, x_test, y_train,  y_test = train_test_split(data1,  Y, test_size=0.3, random_state=42, shuffle=True)

# 创建LightGBM分类器
lgb_classifier = lgb.LGBMClassifier()
# gbdt_classifier =

# 定义超参数空间
param_grid = {
    'learning_rate': [i for i in np.arange(0.09,0.11,0.001)],
    # 'max_depth': [5],
    # 'num_leaves': [27],
    'n_estimators': [i for i in range(450,550,1)]
}


search_space = {
    'lgb__learning_rate': UniformDistribution(low=0.01, high=0.1, q=0.01),
    'lgb__max_depth': UniformDistribution(low=2, high=8, q=1),
    'lgb__num_leaves': UniformDistribution(low=10, high=100, q=1),
    'lgb__n_estimators': UniformDistribution(low=50, high=500, q=50)
}
# 使用GridSearchCV进行网格搜索
grid_search = GridSearchCV(lgb_classifier, param_grid, cv=5)
grid_search.fit(x_train, y_train)


# # 使用遗传算法进行特征选择
# # 定义遗传搜索对象
# from sklearn.pipeline import Pipeline
# estimator = LGBMClassifier(random_state=42)
# pipe = Pipeline(steps=[('lgb', lgb_classifier)])
# genetic_search = GeneticSearchCV(pipe, search_space)
#
# # 进行超参数搜索
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# # print(len(np.array(x_train)))
# # print(np.array(y_train))
# genetic_search.fit(x_train, y_train)
#
# # 输出最优超参数组合和其在测试集上的准确率
# print('Best parameters:', genetic_search.best_params_)
# y_pred = genetic_search.predict(x_test)
y_pred = grid_search.predict(x_test)
print('Test accuracy:', accuracy_score(y_test, y_pred))
score = accuracy_score(y_test, y_pred)
Filepath = 'result.txt'
with open(Filepath,'a') as filewrite:
    filewrite.write('Best parameters:'+ str(grid_search.best_params_))
    filewrite.write('Test accuracy:' + str(score))


# 输出最佳参数
# print('Best parameters using grid search: \n', grid_search.best_params_)
# print('Best parameters using genetic search: \n', genetic_selector.best_params_)

