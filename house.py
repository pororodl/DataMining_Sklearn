import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def read_house():
    data = pd.read_csv('./house/house-votes-84.data',names=['ClassName','handicapped-infants',
                                                            'water-project-cost-sharing','adoption-of-the-budget-resolution',
                                                            'physician-fee-freeze','el-salvador-aid','religious-groups-in-schools',
                                                            'anti-satellite-test-ban','aid-to-nicaraguan-contras',
                                                            'mx-missile','immigration',
                                                            'synfuels-corporation-cutback','education-spending',
                                                            'superfund-right-to-sue','crime',
                                                            'duty-free-exports','export-administration-act-south-africa'])
    # print(data.head())
    col_list = data.columns.values.tolist()   # 获取列索引的名称  如果获取行索引的名称：row_list = data._stat_axis.values.tolist()

    # print(col_list)
    X_mapping = {'n': 0, 'y':1,'?':2}
    for colname in col_list:
        if colname != 'ClassName':
            data[colname] = data[colname].map(X_mapping)
    X = data.iloc[:,1:8]
    # X = data.iloc[:, 1:17]
    y_mapping = {'republican':0,'democrat':1}
    data['ClassName'] = data['ClassName'].map(y_mapping)
    y = data['ClassName']
    # print(y.head())
    y = label_binarize(y, classes=[0, 1])

    return X,y

if __name__ =='__main__':
    # 获取数据
    X,y =read_house()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)
    knn = KNeighborsClassifier()
    tree = tree.DecisionTreeClassifier()
    random_forest = RandomForestClassifier(n_estimators=8)
    neural_network = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                                   solver='adam', alpha=0.0001, max_iter=200)
    svc = SVC()
    models = {'knn': knn, 'tree': tree, 'random_forest': random_forest,'neural_network':neural_network,'SVM':svc}
    time_result = []
    model_result = []
    train_score_result = []
    test_score_result = []
    for m in models:
        start_time = time.time()
        print(start_time)
        # print(models[m])
        models[m].fit(X_train, y_train)
        train_score = models[m].score(X, y)
        train_score = round(train_score,4)
        print('Traing train_score:', train_score)
        end_time = time.time()
        cost_time = end_time - start_time
        cost_time = round(cost_time,3)
        time_result.append(cost_time)
        model_result.append(m)
        train_score_result.append(train_score)
        print('CostTime:', cost_time)
        test_score = models[m].score(X_test, y_test)
        test_score = round(test_score, 4)
        print('Test Accurancy', test_score)
        test_score_result.append(test_score)
        print('---------------------------------------------------------')

# 可视化
# 对所用时间的可视化
plt.figure(figsize=(15,15))
plt.subplot(121)
rects = plt.bar(model_result,time_result,width=0.3,align='center')

plt.xlabel('ClassifierModel')
plt.ylabel('CostTime(sec)')

for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2,height,str(height),ha='center',va='bottom')
plt.title('cost time of training house——vote')

# 对准确率的可视化
n = np.arange(len(model_result))
fig, ax = plt.subplots(figsize=(15,15))
rect1 = ax.bar(n-0.15, train_score_result,width=0.3,tick_label=model_result ,align='center')
rect2 = ax.bar(n+0.15, test_score_result,width=0.3,align='center',color = 'yellow')
plt.xlabel('ClassifierModel')
plt.ylabel('Accurancy')
plt.legend(['Train score','Test score'],loc=1)
for rect in rect1+rect2:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2,height,str(height*100)+'%',ha='center',va='bottom')
plt.title('Accurancy of house_vote')
# plt.bar()
plt.show()