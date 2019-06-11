import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

names = ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean', 'radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 
dataset = pd.read_csv('breast_cancer1.csv',names = names)
dataset['diagnosis'] = dataset['diagnosis']\
.map({'M':1, 'B':0})
ano=dataset.loc[dataset['diagnosis']==1]
nor=dataset.loc[dataset.diagnosis==0]

ano = ano.drop('id_number',1)
nor = nor.drop('id_number',1)

X_train = nor.loc[0:300,:]
X_train = X_train.drop('diagnosis',1)

X_test1 = nor.loc[300:358,:]
X_test2 = ano.loc[:,:]
X_test = X_test1.append(X_test2)
X_test = X_test.drop('diagnosis',1)

y_test1 = nor.loc[300:358,'diagnosis']
y_test2 = ano.loc[:,'diagnosis']
y_test = y_test1.append(y_test2)
y_test.reset_index(inplace = True,drop = True)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.svm import OneClassSVM
classifier = OneClassSVM(kernel='rbf',nu=0.01,gamma=0.08)
classifier.fit(X_train)
y_pred = classifier.predict(X_test)

TP = FN = FP = TN = 0
for j in range(len(y_test)):
    if y_test[j]== 0 and y_pred[j] == 1:
        TP = TP+1
    elif y_test[j]== 0 and y_pred[j] == -1:
        FN = FN+1
    elif y_test[j]== 1 and y_pred[j] == 1:
        FP = FP+1
    else:
        TN = TN +1
print (TP,  FN,  FP,  TN)

accuracy = (TP+TN)/(TP+FN+FP+TN)
print (accuracy)
sensitivity = TP/(TP+FN)
print (sensitivity)
specificity = TN/(TN+FP)
print (specificity)

'''
parameters = [{'nu':[0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]},{'kernel':['rbf','linear']},{'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train)

print(grid_search.best_params_)
'''

