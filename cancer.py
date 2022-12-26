# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 22:04:24 2022

@author: 12345
"""

#匯入
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#新增資料集
my_data = load_breast_cancer()

#檢視資料集
print("資料集: "+str(my_data)+"\n\n")

#觀察資料集
print("資料集的key: "+str(my_data.keys())+"\n")
print("資料筆數: "+str(my_data.data.shape)+"\n")
print("第一筆資料內容: "+str(my_data.data[0])+"\n")
print("第一筆的分類目標: "+str(my_data.target[0])+"\n")
print("預測目標的種類: "+str(my_data.target_names)+"\n\n")

#建立測試集和訓練集
train_x, test_x, train_y, test_y = train_test_split(my_data.data, my_data.target, test_size= 0.2, random_state=18, shuffle=True)

#驗證測試集和訓練集
print("原始資料維度大小: "+str(my_data.data.shape)+"\n")
print("訓練集維度大小: "+str(train_x.shape)+"\n")
print("測試集維度大小: "+str(test_x.shape)+"\n\n")

#訓練模型
my_model = KNeighborsClassifier(n_neighbors=6)   #當K值為6時準確率最高 (95.6%)
my_model.fit(train_x, train_y)
print("模型訓練完成!\n\n")

#模型評分
test_score = my_model.score(test_x, test_y)
print("模型預估準確度: "+str(test_score)+"\n\n")