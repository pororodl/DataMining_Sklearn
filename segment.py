import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def read_segment():
    traindata = pd.read_csv('./segment/s_train.txt',names=['region-centroid-col','region-centroid-row',
                                                      'region-pixel-count','short-line-density-5',
                                                      'short-line-density-2','vedge-mean',
                                                      'vegde-sd','hedge-mean','hedge-sd',
                                                      'intensity-mean','rawred-mean','rawblue-mean',
                                                      'rawgreen-mean','exred-mean','exblue-mean',
                                                      'exgreen-mean','value-mean','saturatoin-mean',
                                                      'hue-mean','classes'])
    # print(traindata.head())

    testdata = pd.read_csv('./segment/s_test.txt', names=['region-centroid-col', 'region-centroid-row',
                                                            'region-pixel-count', 'short-line-density-5',
                                                            'short-line-density-2', 'vedge-mean',
                                                            'vegde-sd', 'hedge-mean', 'hedge-sd',
                                                            'intensity-mean', 'rawred-mean', 'rawblue-mean',
                                                            'rawgreen-mean', 'exred-mean', 'exblue-mean',
                                                            'exgreen-mean', 'value-mean', 'saturatoin-mean',
                                                            'hue-mean', 'classes'])
    print(testdata.head())
    X = traindata.iloc[:,0:19]
    X_test = testdata.iloc[:,0:19]
    y_mapping = {"brickface":0,"sky":1,"foliage":2,"cement":3,"window":4,"path":5,"grass":6}
    traindata['classes'] = traindata['classes'].map(y_mapping)
    testdata['classes'] = testdata['classes'].map(y_mapping)
    y = traindata['classes']
    y_test = testdata['classes']
    print(y.head())
    print(y_test.head())
    y = label_binarize(y,classes=[0,1,2,3,4,5,6])
    y_test = label_binarize(y_test,classes=[0,1,2,3,4,5,6])
    # print(y[:3])
    return X,y,X_test,y_test


if __name__ =='__main__':
    # 获取数据
    X, y, X_test,y_test = read_segment()

    knn = KNeighborsClassifier()
    tree = tree.DecisionTreeClassifier()
    random_forest = RandomForestClassifier(n_estimators=8)
    neural_network = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',
                                   solver='adam',alpha=0.0001,max_iter=200)
    models = {'knn':knn,'tree':tree,'random_forest':random_forest,
              'neural_network':neural_network}
    time_result = []
    model_result = []
    for m in models:
        start_time = time.time()
        print(start_time)
        # print(models[m])
        models[m].fit(X,y)
        print('Traing Accurancy:',models[m].score(X,y))
        end_time = time.time()
        cost_time = end_time-start_time
        time_result.append(cost_time)
        model_result.append(m)
        print('CostTime:',cost_time)

        print('Test Accurancy',models[m].score(X_test,y_test))
        print('---------------------------------------------------------')

plt.bar(model_result,time_result)
# plt.bar()
plt.xlabel('ClassifierModel')
plt.ylabel('CostTime(sec)')
plt.title('house_votes')
# plt.legend()
plt.show()








