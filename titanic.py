# -*- coding: utf-8 -*-
"""


@author: 12345
"""

#匯入
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
import pandas as pd

#匯入 csv 檔
data = pd.read_csv('train.csv')

#資料轉換
data.loc[data['Sex']=='male', 'Sex'] = 0   #把male換成0
data.loc[data['Sex']=='female', 'Sex'] = 1 #把female換成1


""""
資料分析
"""

#存活者性別
data_sur = data[data['Survived']==1]
df_male = data_sur[data_sur['Sex']==0]
df_female = data_sur[data_sur['Sex']==1]
df = pd.DataFrame({'Sex':['df_male', 'df_fenake'], 'val':[len(df_male), len(df_female)]})
ax = df.plot.bar(x='Sex', y='val', rot=0)   #畫圖
#存活者年紀
df_young = data_sur[data_sur['Age']<=30]
df_older = data_sur[data_sur['Age']>30]
df = pd.DataFrame({'Age':['young', 'older'], 'val':[len(df_young), len(df_older)]})
ax = df.plot.bar(x='Age', y='val', rot=0)   #畫圖

"""
決策樹製作
"""

#資料整理
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])

#資料轉換
typeEmbarked = list(set(data['Embarked']))
for i in range(len(typeEmbarked)):
    print(typeEmbarked[i])
    row = data['Embarked'] == typeEmbarked[i]
    data.loc[row, 'Embarked'] = i
    
typeSex = list(set(data['Sex']))
for i in range(len(typeSex)):
    print(typeSex[i])
    rows = data['Sex'] == typeSex[i]
    data.loc[rows, 'Sex'] = i
    
#缺失值處理
data = data.fillna(999)

#訓練模型
X_train = data[:750]
X_test = data[750:]
y_train = X_train.pop('Survived')

#建立並訓練決策樹
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

#評估準確率
y_test = X_test.pop('Survived')
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))   #準確率
print(recall_score(y_test, y_pred))     #召回率
