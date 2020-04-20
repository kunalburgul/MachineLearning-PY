#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:04:31 2019

@author: pymacbit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################################

# Logistic Regression

# Importing the dataset_LoR 
dataset_LoR = pd.read_csv("Social_Network_Ads.csv")
X = dataset_LoR.iloc[:, [2,3]].values
y = dataset_LoR.iloc[:, 4].values

# Splitting the dataset_LoR into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

############################################################################

# KNN Classifier

# Importing the dataset_KNN 
dataset_KNN = pd.read_csv("Social_Network_Ads.csv")
X = dataset_KNN.iloc[:, [2,3]].values
y = dataset_KNN.iloc[:, 4].values

# Splitting the dataset_KNN into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the KNN Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p = 2)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

###########################################################################

# SVM Intution

# Importing the dataset_SVM 
dataset_SVM = pd.read_csv("Social_Network_Ads.csv")
X = dataset_SVM.iloc[:, [2,3]].values
y = dataset_SVM.iloc[:, 4].values

# Splitting the dataset_KNN into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the SVM Intution to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('SVM Intution (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop=  X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('SVM Intution (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

###########################################################################

# Kernel SVM

# Importing the dataset_K_SVM 
dataset_K_SVM = pd.read_csv("Social_Network_Ads.csv")
X = dataset_K_SVM.iloc[:, [2,3]].values
y = dataset_K_SVM.iloc[:, 4].values

# Splitting the dataset_K_SVM into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the SVM Intution to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop=  X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

##########################################################################

# Naive Bayes

# Importing the dataset_NB
dataset_NB = pd.read_csv("Social_Network_Ads.csv")
X = dataset_NB.iloc[:, [2,3]].values
y = dataset_NB.iloc[:, 4].values

# Splitting the dataset_NB into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop=  X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

###########################################################################

# Desion Tree Classifier

# Importing the dataset_DTC
dataset_DTC = pd.read_csv("Social_Network_Ads.csv")
X = dataset_DTC.iloc[:, [2,3]].values
y = dataset_DTC.iloc[:, 4].values

# Splitting the dataset_DTC into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Desion Tree Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Desion Tree Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop=  X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Desion Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

###########################################################################

# Random Forest Classifier

# Importing the dataset_RFC
dataset_RFC = pd.read_csv("Social_Network_Ads.csv")
X = dataset_RFC.iloc[:, [2,3]].values
y = dataset_RFC.iloc[:, 4].values

# Splitting the dataset_RFC into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy' ,random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop= X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 =np.meshgrid(np.arange(start =X_set[:,0].min()-1, stop= X_set[:, 0].max()+1,step = 0.01),
                    np.arange(start =X_set[:,1].min()-1, stop=  X_set[:, 1].max()+1,step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.xlim(X1.min(),X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1],
                c= ListedColormap(('red','green'))(i),label = j)
plt.title('Random Forest Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()