# -*- coding: utf-8 -*-
"""
@Created on 2018/5/21 16:51

@author: ZhifengFang
"""
import pandas as pd
import numpy as np

data=pd.read_csv('train.csv')
X=data.iloc[:,1:-1]
y=data.iloc[:,-1]
test=pd.read_csv('test.csv')
sub=pd.read_csv('sample_submit.csv')
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(X,y)
y_pred=classifier.predict()