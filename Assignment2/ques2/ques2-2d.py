#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Data from the CSV file
#get_ipython().run_line_magic('matplotlib', 'inline')
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.svm import SVC as svc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import time
from scipy import spatial
from itertools import combinations 
import pickle


# 28*28 - one row size = 784 - one training data

# In[2]:


def input_read(file, sep, class0=None, class1=None):
    myFile = pd.read_csv(file, sep=sep, header=None, index_col=False)
    if ((class0!=None) and (class1!=None)):
        df_train = myFile[(myFile.get(784) == class0) | (myFile.get(784) == class1)]
    else: df_train = myFile
    train_classes = df_train[784]
    if ((class0!=None) and (class1!=None)):
        train_classes = (train_classes.apply(lambda x: 1 if (x==class1) else -1))*1.
    train_classes.index = np.arange(0, len(df_train))
    df_train = df_train.drop(784, axis=1)
    df_train.index=(np.arange(0, len(df_train)))
    df_train /= 255
    return df_train.to_numpy(), train_classes.to_numpy();


# In[3]:


X_train, train_classes = input_read('fashion_mnist/train.csv', ',')
X_val, val_classes = input_read('fashion_mnist/val.csv', ',')
X_test, test_classes = input_read('fashion_mnist/test.csv', ',')


# In[4]:


X_all = np.vstack((X_train, X_val))
classes_xall = np.hstack((train_classes, val_classes)).reshape(-1, 1)
temp = np.hstack((X_all, classes_xall))
np.random.shuffle(temp)
fold1, fold2, fold3, fold4, fold5 = np.split(temp, 5)
fold = [fold1, fold2, fold3, fold4, fold5]
c= [1e-5 , 1e-3 , 1, 5, 10]


# In[5]:


from itertools import combinations
K=5
x = np.arange(K)

train_data = []
val_data = []
comb = [i for i in combinations(x,(K-1))]
print("The combinations of train and val data for 5-fold cross validations are")
for i in comb:
    t = [fold[x] for x in i]
    train_data.append(np.vstack((t[0], t[1], t[2], t[3])))
    rem= np.setdiff1d(x, i)
    val_data.append(fold[rem[0]])
    print("Train Sets = {} Val Set = {}".format(i, rem[0]))


# In[6]:


train_class=[]
val_class=[]
for i in range(len(train_data)):
    train_class.append(train_data[i][:,-1])
    train_data[i]=train_data[i][:,:-1]
    
    val_class.append(val_data[i][:,-1])
    val_data[i]=val_data[i][:,:-1]


# In[8]:


def sklearn_svm(X_train, train_classes, shape, gamma='scale', c=1):
    start = time.time()
    svc_classifier = svc(C=c, kernel=shape, gamma=gamma)
    svc_classifier.fit(X_train, train_classes)
    print("The time taken to train SVM model using SVM classifier SKLEARN and {} Kernel = {:2.3f}sec"
          .format(shape, time.time()-start))    
    return svc_classifier


# In[ ]:


print("Running SVM Classifier from SKLEARN to classify with Gaussian Kernel")
print("--------------------TRAINING--------------------------------------------")
gamma=0.05
train_acc_c=[];val_acc_c=[];test_acc_c =[]; test_accuracy=[]
index_best_model =[]

val_pred_fold = []
test_pred_fold = []

c= [1e-5, 1e-3]

for i in range(len(c)):
    train_pred = []; val_pred =[]; test_pred=[]; svc_classifier=[]
    print("\n\n\n@@@@@@@@@@-----------Training for C={}----------------@@@@@@@@@@@\n".format(c[i]))
    for j in range(len(train_data)):
        print("\n------------Training SVM on train/val data from combination number {} --------------\n".format(j))
        svc_classifier.append(sklearn_svm(train_data[j], train_class[j], 'rbf', gamma, c[i]))
        train_acc_svc = svc_classifier[j].score(train_data[j], train_class[j])
        val_acc_svc = svc_classifier[j].score(val_data[j], val_class[j])
        test_acc_svc = svc_classifier[j].score(X_test, test_classes)
        
        train_pred.append(train_acc_svc)
        val_pred.append(val_acc_svc)
        test_pred.append(test_acc_svc)
        val_pred_fold.append(val_acc_svc)
        test_pred_fold.append(test_acc_svc)
    
    test_acc_c.append(sum(test_pred)/len(test_pred))
    val_acc_c.append(sum(val_pred)/len(val_pred))
    train_acc_c.append(sum(train_pred)/len(train_pred))
    index_best_model.append(np.where(val_pred == np.max(val_pred))[0][0])
    test_accuracy.append(svc_classifier[index_best_model[i]].score(X_test, test_classes))

    
    print("\nBest classifier for C= {} found on the fold number {} with val acc = {:2.3f}% and test acc = {:2.3f}%"
          .format(c[i], index_best_model[i], val_pred[index_best_model[i]]*100, test_pred[index_best_model[i]]*100))
    print("The average train accuracy for C= {} is = {:2.3f}%".format(c[i], train_acc_c[i]*100))
    print("The average validation accuracy for C= {} is = {:2.3f}%".format(c[i], val_acc_c[i]*100))
    
    print("The test accuracy on best classifier for C= {} is = {:2.3f}%".format(c[i], test_acc_c[i]*100))    


# In[ ]:


print("---------------------Accumulated results---------------------------")


print("\nValidation Accuracy for all the folds\n")
print(val_pred_fold)
print("\nTest Accuracy for all the folds\n")
print(test_pred_fold)

print("\nAverage Validation Accuracy for different C\n")
print(val_acc_c)

print("\nAverage Test Accuracy for different C\n")
print(test_acc_c)

print("\n Test Accuracy on Best classifier for different C\n")
print(test_accuracy)

print("\nAverage Train Accuracy for different C\n")
print(train_acc_c)

print("\nBest Classifier Fold for different C\n")
print(index_best_model)

