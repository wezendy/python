#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:04:29 2022

@author: jakabbernat
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
import pandas as pd;  # importing pandas data analysis tool
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.tree import DecisionTreeClassifier; 
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.neural_network import MLPClassifier;
from sklearn.cluster import KMeans;
from sklearn.metrics import davies_bouldin_score;
from sklearn.decomposition import PCA;
from sklearn import metrics;
from sklearn.model_selection import train_test_split
import seaborn as sns;  # importing statistical data visualization

#1.feladat
url = 'https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/labor_exercise_wednesday1.csv';
raw_data = urlopen(url);  # reading the first row with attribute names
data = np.loadtxt(raw_data, skiprows=1, delimiter=",");  # reading numerical data from csv file
del raw_data;

#2. feladat
X = data[:,0:10]
y = data[:,10]
df = pd.DataFrame(data=data,columns=['Var1', 'Var2', 'Var3','Var4',
                                     'Var5','Var6','Var7','Var8','Var9','Var10','target'])

print('Number of records: ',df.size)
print('Number of attributes: ', len(df.columns.drop('target')))
print('Number of classes: ',len(df.groupby(by='target').size()))

grouped_by_target = df.groupby(by='target');  # grouping by target

mean_by_target = grouped_by_target.mean();
corr_by_target = grouped_by_target.corr();

print('MEAN')
print(mean_by_target) #átlag

print('CORR')
print(corr_by_target) #korreláció

#3. feladat

plt.figure(2);
pd.plotting.andrews_curves(df,class_column='target',color=['blue','red']);
plt.show();

sns.set(); 

colors = ['blue','red'];
sns.relplot(data=df, x='Var1', y='Var3', 
            hue='target', palette=colors);

#4. feladat

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=100)

#5. feladat

class_tree = DecisionTreeClassifier(criterion = 'entropy',max_depth = 5);

class_tree.fit(X_train, y_train);
score_train_tree = class_tree.score(X_train, y_train);
score_test_tree = class_tree.score(X_test, y_test);


logreg_classifier = LogisticRegression(solver = 'sag');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);
score_test_logreg = logreg_classifier.score(X_test,y_test);
ypred_logreg = logreg_classifier.predict(X_test);
yprobab_logreg = logreg_classifier.predict_proba(X_test);

neural_classifier = MLPClassifier(hidden_layer_sizes=(1,2),
                                  activation='logistic',
                                  max_iter=1000);
neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);
score_test_neural = neural_classifier.score(X_test,y_test);
ypred_neural = neural_classifier.predict(X_test);
yprobab_neural = neural_classifier.predict_proba(X_test);

print(f'Test score of tree in %: {score_test_tree*100}');
print(f'Test score of logreg in %: {score_test_logreg*100}'); 
print(f'Test score of neural in %: {score_test_neural*100}');

#6. feladat

"""
#???
#ha csak a blokkot futtatom, mukodik
labelName = 'logreg'
fpr, tpr, _ = metrics.roc_curve(y_test, yprobab_logreg[:,0], pos_label=0)
roc_auc = metrics.auc(fpr, tpr)
"""

maxScore = max(score_test_tree, score_test_neural, score_test_logreg)
if maxScore == score_test_neural:
    labelName = 'neural'
    cm = metrics.confusion_matrix(y_test, ypred_neural)

    fpr, tpr, _ = metrics.roc_curve(y_test, yprobab_neural[:,0], pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)
if maxScore == score_test_logreg:
    labelName = 'logreg'
    cm = metrics.confusion_matrix(y_test, ypred_logreg)

    fpr, tpr, _ = metrics.roc_curve(y_test, yprobab_logreg[:,0], pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)

plt.figure(3);
plt.plot(fpr, tpr, color='red',
         lw=2, label=f'{labelName}(area = %0.2f)' % roc_auc);
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

#7.feladat

without_letter = df.get(['Var1', 'Var2', 'Var3','Var4',
                                     'Var5','Var6','Var7','Var8','Var9','Var10'])

kmeans2 = KMeans(n_clusters=2, random_state=100);

kmeans2.fit(without_letter);
labels2 = kmeans2.labels_;
centers2 = kmeans2.cluster_centers_;
DB2 = davies_bouldin_score(without_letter,labels2);

print(f'2 -es Klaszterrel a DB : {DB2}')

pca = PCA(n_components=2);
pca.fit(without_letter.values);
data_pc = pca.transform(without_letter.values);
centers_pc = pca.transform(centers2);

kmeans4 = KMeans(n_clusters=4, random_state=100);

kmeans4.fit(without_letter);
labels4 = kmeans4.labels_;
centers4 = kmeans4.cluster_centers_;
DB4 = davies_bouldin_score(without_letter,labels4);

print(f'4 -es Klaszterrel a DB : {DB4}')

pca = PCA(n_components=4);
pca.fit(without_letter.values);
data_pc = pca.transform(without_letter.values);
centers_pc = pca.transform(centers4);

Max_K = 30;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
optimal_cluster = 0
min_score = 10000
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=100);
    kmeans.fit(without_letter);
    bc_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(without_letter,bc_labels);

# Visualization of SSE values    
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();
    
fig = plt.figure(4);
plt.title('Clustering of the Letter data after PCA (2)');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(data_pc[:,0],data_pc[:,1],s=50,c=labels2);
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');
plt.show();

fig = plt.figure(5);
plt.title('Clustering of the Letter data after PCA (4)');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(data_pc[:,0],data_pc[:,1],s=50,c=labels4);
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');
plt.show();