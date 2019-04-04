import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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
    X_mapping = {'n': 0, 'y':1,'?':0}
    for colname in col_list:
        if colname != 'ClassName':
            data[colname] = data[colname].map(X_mapping)
    X = data.iloc[:,1:17]

    y_mapping = {'republican':0,'democrat':1}
    data['ClassName'] = data['ClassName'].map(y_mapping)
    y = data['ClassName']
    # print(y.head())
    y = label_binarize(y, classes=[0, 1])

    return X,y

if __name__ =='__main__':
    # 获取数据
    X,y =read_house()
    knn = KNeighborsClassifier()
    tree = tree.DecisionTreeClassifier()
    random_forest = RandomForestClassifier(n_estimators=8)
    models = {'knn': knn, 'tree': tree, 'random_forest': random_forest}
    time_result = []
    model_result = []
    for m in models:
        start_time = time.time()
        print(start_time)
        # print(models[m])
        models[m].fit(X, y)
        print('Traing Accurancy:', models[m].score(X, y))
        end_time = time.time()
        cost_time = end_time - start_time
        time_result.append(cost_time)
        model_result.append(m)
        print('CostTime:', cost_time)

        # print('Test Accurancy',models[m].score(X_test,y_test))
        print('---------------------------------------------------------')

plt.bar(model_result, time_result)
# plt.bar()
plt.xlabel('ClassifierModel')
plt.ylabel('CostTime(sec)')
plt.title('house_votes')
# plt.legend()
plt.show()
