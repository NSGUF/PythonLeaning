# -*- coding: utf-8 -*-
"""
@Created on 2018/5/21 15:43

@author: ZhifengFang
"""
import pandas as pd
import numpy as np

def load_info(filename):
    data = pd.read_csv(filename)
    X = np.array(data.iloc[:, 1:-1])
    y = np.array(data.iloc[:, -1])
    return X, y
def load_test(filename):
    data=pd.read_csv(filename)
    index=np.array(data.iloc[:,0])
    X=np.array(data.iloc[:,1:])
    return index,X

'''
X, y = load_info('train.csv')
num_train = int(len(X) * 0.8)
X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]
from sklearn import linear_model

re = linear_model.LinearRegression()
re.fit(X_train, y_train)
y_pre = re.predict(X_test)
print(X)
'''
import pickle as p
import sklearn.metrics as sm
def save_model(regressor):
    with open('model.pkl','wb') as f:
        p.dump(regressor,f)
def load_model(inputfile):
    with open(inputfile,'rb') as f:
        regression=p.load(f)
    return regression
'''
X,y=load_info('train.csv')
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=2)
rf_regressor.fit(X, y)
save_model(rf_regressor)
print(sm.mean_absolute_error(y_test,y_pre))
'''
rf_regressor=load_model('model.pkl')
index,X_test=load_test('test.csv')

y_pre = rf_regressor.predict(X_test)

id=pd.DataFrame(index,columns=['id'])
y=pd.DataFrame(y_pre,columns=['y'])

result=pd.concat([id,y],axis=1)
result.to_csv('sample_submit.csv', index=False)
