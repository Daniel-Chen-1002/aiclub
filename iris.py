# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 23:03:38 2022

@author: 12345
"""

#匯入資料
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#觀察資料集
iris = load_iris()
print(iris.keys())
print("資料筆數: "+str(iris.data.shape)+"\n")
print("資料欄位: "+str(iris.feature_names)+"\n")
print("第一筆資料: "+str(iris.data[0])+"\n")
print("第一筆預測目標: "+str(iris.target[0])+"\n\n")

#繪製散點圖
scatter = plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
#plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names.tolist())

scatter = plt.scatter(iris.data[:,2], iris.data[:,3], c=iris.target)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
#plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names.tolist())

#訓練模型
estimator = KMeans(n_clusters=3, random_state=54)
estimator.fit(iris.data)
print(estimator.labels_)

#模型作圖
scatter = plt.scatter(iris.data[:,2], iris.data[:,3], c=estimator.labels_)
plt.xlabel('mPetal length')
plt.ylabel('mPetal width')

plt.scatter(estimator.cluster_centers_[:,2], estimator.cluster_centers_[:,3], marker='*', c='red', s=100)
#plt.legend( *scatter.legend_elements())
plt.show()

#3D 作圖
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=130)
ax.set_zlim(2, 4.5)

ax.scatter(iris.data[:,2], iris.data[:,3], iris.data[:,1], c=estimator.labels_)
ax.scatter(estimator.cluster_centers_[:,2], estimator.cluster_centers_[:,3], estimator.cluster_centers_[:,1], marker='*', c='red', s=100)
plt.legend( *scatter.legend_elements(), loc='lower right')
plt.show()