# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:12:13 2017

@author: ZhifengFang
"""
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
 
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=[1,2,3,4,5,6,7,8,9]
y_train=[1,2,3,4,5,6,7,8,9]
#x_test=input_variables_values_test_datasets
 
# Create linear regression object
linear = linear_model.LinearRegression()
 
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
 
#Equation coefficient and Intercept
print('Coefficient: n', linear.coef_)
print('Intercept: n', linear.intercept_)
 
#Predict Output
#predicted= linear.predict(x_test)