# -*- coding: utf-8 -*-
"""
@Created on 2018/4/28 22:49

@author: ZhifengFang
"""
import numpy as np
import pandas as pd
import datetime


def load_data(input_file):
    X = []
    y = []
    data = pd.read_csv(input_file, na_values=0)
    X = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])
    X[X == 'Low'] = 0
    X[X == 'Medium'] = 1
    X[X == 'High'] = 2
    for i in range(X.shape[0]):
        date = X[i, 2].split('/')  # 11/9/90
        d1 = datetime.date(2018, 4, 30)
        if np.int(date[2])<18:
            year = 2000 + np.int(date[2])
        else:
            year = 1900 + np.int(date[2])
        d2 = datetime.date(year, np.int(date[0]), np.int(date[1]))
        X[i, 2] = (d1 - d2).total_seconds() / 60 / 60 / 24

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (np.isnan(X[i, j])):
                X[i, j] = 0
    return X, y


def load_test_data(input_file):
    data = pd.read_csv(input_file)
    X = np.array(data.iloc[:, 1:])
    index = np.array(data.iloc[:, 0])
    X[X == 'Low'] = 0
    X[X == 'Medium'] = 1
    X[X == 'High'] = 2
    for i in range(X.shape[0]):
        date = X[i, 2].split('/')  # 11/9/90
        d1 = datetime.date(2018, 4, 30)

        if np.int(date[2])<18:
            year = 2000 + np.int(date[2])
        else:
            year = 1900 + np.int(date[2])
        d2 = datetime.date(year, np.int(date[0]), np.int(date[1]))
        X[i, 2] = (d1 - d2).total_seconds() / 60 / 60 / 24

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (np.isnan(X[i, j])):
                X[i, j] = 0
    return X, index


input_file = 'train.csv'

X, y = load_data(input_file)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_training = np.array(X[0:num_training])
Y_training = np.array(y[0:num_training])

X_test = np.array(X[num_training:])
Y_test = np.array(y[num_training:])

'''
from sklearn.tree import DecisionTreeRegressor
dt_regressor=DecisionTreeRegressor(max_depth=20)
dt_regressor.fit(X_training,Y_training)

# X_test,index=load_test_data('test.csv')
y_dt_pred=dt_regressor.predict(X_test)
'''
'''
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=2)
rf_regressor.fit(X_training, Y_training)
y_pre = rf_regressor.predict(X_test)
'''
import pickle  as p
def save_model(regressor):
    with open('saved_model_output.pkl', 'wb') as f:
        p.dump(regressor, f)

def load_model(input_file):
    with open(input_file, 'rb') as f:
        model_liner = p.load(f)
    return model_liner

# save_model(rf_regressor)

import sklearn.metrics as sm
rf_regressor=load_model('saved_model_output.pkl')
y_pre = rf_regressor.predict(X_test)
print('平均绝对误差：', sm.mean_absolute_error(Y_test, y_pre))
# X_test, index = load_test_data('test.csv')
# y_dt_pred = rf_regressor.predict(X_test)
# # print('平均绝对误差：', sm.mean_absolute_error(Y_test, y_pre))
#
# id = pd.DataFrame(index, columns=['id'])
# y = pd.DataFrame(y_dt_pred, columns=['y'])
# result = pd.concat([id, y], axis=1)
#
# result.to_csv('sample_submit.csv', index=False)  # 写入数据
