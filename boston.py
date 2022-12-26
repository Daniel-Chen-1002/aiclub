# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 21:36:04 2022

@author: 12345
"""

#匯入
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#資料集新增
my_data = load_boston()

#檢視資料集
print("資料集: "+str(my_data)+"\n\n")

#觀察資料集
print("資料集的key: "+str(my_data.keys())+"\n")
print("資料筆數: "+str(my_data.data.shape)+"\n")
print("資料欄位: "+str(my_data.feature_names)+"\n")
print("第一筆資料內容: "+str(my_data.data[0])+"\n")
print("第一筆的預測目標: "+str(my_data.target[0])+"\n\n")

#建立訓練集&測試集
train_x, test_x, train_y, test_y = train_test_split(my_data.data, my_data.target, test_size=0.2, random_state=43, shuffle=True)

#驗證訓練集&測試集
print("原始資料維度大小: "+str(my_data.data.shape)+"\n")
print("訓練集維度大小: "+str(train_x.shape)+"\n")
print("測試集維度大小: "+str(test_x.shape)+"\n\n")

#訓練模型
my_model = LinearRegression()
my_model.fit(train_x, train_y)
print("模型訓練完畢!\n\n")
#測試模型
pred = my_model.predict(test_x)     
score = mean_squared_error(test_y, pred)    #MSE值 越接近0月準
print("模型預估誤差: "+str(score)+"\n\n")