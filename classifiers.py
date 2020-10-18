#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np    
from sklearn.model_selection import KFold  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier   
from sklearn import svm  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  

#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from joblib import dump
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc  
#rom sklearn import cross_validation
#rom sklearn.model_selection import train_test_split

# =============================================================================
# feature_names = [    
#     'area',    
#     'perimeter',    
#     'compactness',    
#     'length of kernel',    
#     'width of kernel',    
#     'asymmetry coefficien',    
#     'length of kernel groove',    
# ]    
# =============================================================================
  
COLOUR_FIGURE = False    
  
# =============================================================================
# def load_csv_data(filename):     
#     dataframe = pd.read_csv(filename)
#     #dataframe = pd.read_csv(url,names=names)
#     array = dataframe.values
#     len1=dataframe.columns.size-1
#     #names1 = dataframe.axes[1]
#     #names = names1[0:12]
#    # print(names)
#     
#     data = array[:,0:len1]
#     #print(X.shape)
#     labels = array[:,len1]
#     return data, labels 

def load_csv_data(filename):    
    dataframe = pd.read_csv(filename)
    #dataframe = pd.read_csv(url,names=names)
    array = dataframe.values
    len1=dataframe.columns.size-1
    #names1 = dataframe.axes[1]
    #names = names1[0:12]
   # print(names)

    
    data = array[:,0:len1]
    #print(X.shape)
    labels = array[:,len1]
    smo = SMOTE(random_state=42)
    data, labels = smo.fit_sample(data, labels)

    labels = label_binarize(y_smo, classes=[1,2,3,4])
#    print(labels)
       
    #print(labels.shape[1])
    return data, labels 

# =============================================================================
    #print(Y)
# =============================================================================
#     data = []    
#     labels = []    
#     datafile = open(filename)    
#     for line in datafile:    
#         fields = line.strip().split('\t')    
#         data.append([float(field) for field in fields[:-1]])    
#         labels.append(fields[-1])     
#     data = np.array(data)    
#     labels = np.array(labels)    
#     return data, labels    
# =============================================================================
      
# =============================================================================
# def accuracy(test_labels, pred_lables):    
#     correct = np.sum(test_labels == pred_lables)   
#     n = len(test_labels)    
#     return float(correct) / n    
# =============================================================================
def accuracy(result_set):  
    ac_score=[]
    ka_score=[]
    roc_score=[]
    pre_score=[]
    rec_score=[]
    f_score=[]
    for result in result_set:
        
        ac=accuracy_score(labels[result[1]], result[0])  #accuracy
        ac_score.append(ac)
        
        ka=cohen_kappa_score(labels[result[1]], result[0])   #kappa
        ka_score.append(ka)
        
        pre=precision_score(labels[result[1]], result[0], average='macro') 
        pre_score.append(pre)
        
        rec=recall_score(labels[result[1]], result[0], average='macro') 
        rec_score.append(rec)
        
        f1=f1_score(labels[result[1]], result[0], average='macro') 
        f_score.append(f1)
        
#        roc=roc_auc_score(labels[result[1]], result[2])
#        roc_score.append(roc)
        
    #print(ac_score)
    print('accuracy:',np.mean(ac_score))
    #print(ka_score)
    print('kappa:',np.mean(ka_score))
    
    print('roc_auc',np.mean(roc_score))
    
    print('precision',np.mean(pre_score))
    print('recall',np.mean(rec_score))
    print('f1_score',np.mean(f_score))
    
#------------------------------------------------------------------------------  
#KNN
#------------------------------------------------------------------------------  
def testKNN(features, labels):  
    kf = KFold(n_splits=10, shuffle=True) 
    kf.get_n_splits(len(features))
    clf = KNeighborsClassifier(n_neighbors=3)    
    result_set = []
    for train, test in kf.split(features):

        y_score = clf.fit(features[train],labels[train]).predict_proba(features[test])  
       
        y_scores=y_score[:,1]
        result=clf.predict(features[test])   
        result_set.append((result, test, y_scores))
    return result_set
    
    
    
    # Compute ROC curve and ROC area for each class
# =============================================================================
#     fpr,tpr,threshold = roc_curve(labels[test], y_score) 
#     roc_auc = auc(fpr,tpr) 
#     
#     lw = 2
#     plt.figure(figsize=(10,10))
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') 
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")  
#     plt.show()
# =============================================================================


  
    

#------------------------------------------------------------------------------  
#logistic regression  
#------------------------------------------------------------------------------  
def testLR(features, labels):  
    kf = KFold(n_splits=10, shuffle=True)    
    kf.get_n_splits(len(features))
    clf = LogisticRegression(solver='liblinear')  
    result_set = []
#    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test,clf.decision_function(features[test])) for train, test in kf.split(features)]    
    for train, test in kf.split(features):
        y_score = clf.fit(features[train],labels[train]).predict_proba(features[test])  
        y_scores=y_score[:,1]
        result=clf.predict(features[test])   
        result_set.append((result, test, y_scores))
    return result_set
 

  
#------------------------------------------------------------------------------  
#Naive Bayes
#------------------------------------------------------------------------------  
def testNaiveBayes(features, labels):  
    kf = KFold(n_splits=10, shuffle=True)    
    kf.get_n_splits(len(features))
    clf = GaussianNB()  
    result_set = []
#    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test,clf.decision_function(features[test])) for train, test in kf.split(features)]  
    for train, test in kf.split(features):
        y_score = clf.fit(features[train],labels[train]).predict_proba(features[test]) #训练
        y_scores=y_score[:,1]
        result=clf.predict(features[test])   #预测
        result_set.append((result, test, y_scores))
    return result_set
  
               
#------------------------------------------------------------------------------  
#-SVM  
#------------------------------------------------------------------------------  
def testSVM(features, labels):  
    kf = KFold(n_splits=10, shuffle=True)    
    kf.get_n_splits(len(features))
    clf = svm.SVC(probability=True, gamma='auto')  
    #result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf.split(features)]    
    #score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    #print(score)  
    result_set = []
    for train, test in kf.split(features):
        y_score = clf.fit(features[train],labels[train]).predict_proba(features[test]) #训练
        y_scores=y_score[:,1]
        result=clf.predict(features[test])   #预测
        result_set.append((result, test, y_scores))
    return result_set
 
  
#------------------------------------------------------------------------------  
#-Decision tree
#------------------------------------------------------------------------------  
def testDecisionTree(features, labels):  
    kf = KFold(n_splits=10, shuffle=True)    
    kf.get_n_splits(len(features))
    clf = DecisionTreeClassifier()  
    #result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf.split(features)]    
    #score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    #print(score)  
    result_set = []
    for train, test in kf.split(features):
        y_score = clf.fit(features[train],labels[train]).predict_proba(features[test]) 
        y_scores=y_score[:,1]
        result=clf.predict(features[test])   
        result_set.append((result, test, y_scores))
    return result_set 
      
#------------------------------------------------------------------------------  
#-Random Forest 
#------------------------------------------------------------------------------  
def testRandomForest(features, labels):  
    kf = KFold(n_splits=10, shuffle=True)    
    kf.get_n_splits(len(features))
    clf = RandomForestClassifier(n_estimators=10)  
    #result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf.split(features)]    
    #score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    #print(score)  
    result_set = []
    for train, test in kf.split(features):
        y_score = clf.fit(features[train],labels[train]).predict_proba(features[test]) 
        y_scores=y_score[:,1]
        result=clf.predict(features[test])   
        result_set.append((result, test, y_scores))
    return result_set
  
      
if __name__ == '__main__':  
#     features, labels = load_csv_data('/Users/yumh/Desktop/重要/第二课课程代码及数据文件/data./data/seeds_dataset.txt')
    features, labels = load_csv_data(r'C:\Users\liutt\Desktop\data for paper\achievement_emotion_training_data(1).csv') 
    #print(features)  
    #print(labels)
      
       
    print('KNN: \r')  
    accuracy(testKNN(features, labels))
    print('\r') 
    
    print('LogisticRegression: \r')  
    accuracy(testLR(features, labels)) 
    print('\r') 
       
    print('GaussianNB: \r')  
    accuracy(testNaiveBayes(features, labels)) 
    print('\r')   
    
    print('Decision Tree: \r')  
    accuracy(testDecisionTree(features, labels))
    print('\r') 
    
    print('Random Forest: \r')  
    accuracy(testRandomForest(features, labels))
    print('\r') 
    
    print('SVM: \r')  
    accuracy(testSVM(features, labels))


# In[ ]:




