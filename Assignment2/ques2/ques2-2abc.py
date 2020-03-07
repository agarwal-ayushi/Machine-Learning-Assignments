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


def calc_accuracy(classes, pred_classes):   
    acc = 0
    for i in range(len(classes)):
        if (classes[i]==pred_classes[i]):
            acc+=1
    acc = (acc/len(classes))*100
    return acc


# In[4]:


def gauss_kernel_cvxopt(X_train, train_classes, c=1):
    sv_pos_ind=[]
    sv_neg_ind=[]
    start =time.time()
    m,n = X_train.shape
    K = np.zeros((m,m))
    gamma = 0.05
    start = time.time()
    pdist = spatial.distance.pdist(X_train, 'sqeuclidean')
    K = np.exp(-1*gamma*spatial.distance.squareform(pdist))
    end = time.time()
    print("Time to Calculate Gaussian Kernel matrix for CVXOPT= {}s".format(end-start))

    P = cvxopt_matrix(np.outer(train_classes, train_classes)*K)
    q = cvxopt_matrix(-1*np.ones((m,1)))
    G1 =-1*np.eye(m)
    G2 =c*np.eye(m)
    G = cvxopt_matrix(np.vstack((G1,G2)))
    h1 = np.zeros(m)
    h2 = np.ones(m)*c
    h = cvxopt_matrix(np.hstack((h1,h2)))
    A = cvxopt_matrix(train_classes.reshape(1,-1)*1.)
    b = cvxopt_matrix(np.zeros(1))
    cvx_solver = cvxopt_solvers.qp(P, q, G, h, A, b)
    print("The time taken to optimize SVM model using CVXOPT and Gaussian Kernel = {}sec".format(time.time()-start))    

    return cvx_solver;


# In[5]:


def train_classifiers(X_train, train_classes, class1, class2):
    
    train_classes = train_classes.reshape(-1,1)
    m,n = X_train.shape
    c=1 #penalty weight
    print("Running CVX Optimizer to classify Class {} and Class {} with Gaussian Kernel".format(class1, class2))
    print("--------------------TRAINING--------------------------------------------")
    
    cvx_solver = gauss_kernel_cvxopt(X_train, train_classes, c)
    return cvx_solver


# In[6]:


def predict(X_train, X_val, X_test, train_classes, val_classes, test_classes, w, b, kernel_shape, supp_vec_ind, alpha):
    if(kernel_shape == 'linear'):
        train_pred = [1.0 if x >=0 else -1. for x in (np.dot(X_train,w) + b) ]
        train_acc = calc_accuracy(train_classes, train_pred)

        val_pred = [1. if x >=0 else -1. for x in (np.dot(X_val,w) + b) ]
        val_acc = calc_accuracy(val_classes, val_pred)

        test_pred = [1. if x >=0 else -1. for x in (np.dot(X_test,w) + b) ]
        test_acc = calc_accuracy(test_classes, test_pred)
        
    elif(kernel_shape == 'gaussian'):
        cdist = spatial.distance.cdist(X_train[supp_vec_ind], X_train, 'sqeuclidean')
        K_train = np.exp(-1*0.05*(cdist))
        w = np.dot(K_train.T, (alpha[supp_vec_ind]*train_classes[supp_vec_ind]))
        train_pred = w + b
        
        cdist = spatial.distance.cdist(X_train[supp_vec_ind], X_val, 'sqeuclidean')
        K_val = np.exp(-1*0.05*(cdist))
        w = np.dot(K_val.T, (alpha[supp_vec_ind]*train_classes[supp_vec_ind]))
        val_pred = w + b
        
        cdist = spatial.distance.cdist(X_train[supp_vec_ind], X_test, 'sqeuclidean')
        K_test = np.exp(-1*0.05*(cdist))
        w = np.dot(K_test.T, (alpha[supp_vec_ind]*train_classes[supp_vec_ind]))
        test_pred = w + b
        
    return train_pred, val_pred, test_pred;


# # 50 mins to train classifiers 

# In[7]:


label=range(10)
comb = combinations(label, 2) 
num_classifier=0
for i in list(comb): 
    num_classifier+=1
    
comb = combinations(label, 2) 

cvx_solver = []
print(num_classifier)
start = time.time()
#Training all the classifiers per class pair
for k in list(comb):   
    X_train, train_classes = input_read('fashion_mnist/train.csv', ',', k[0], k[1])
    cvx_solver.append(train_classifiers(X_train, train_classes, k[0], k[1]))
print("Time taken to train all the classifiers (10C2) with CVX OPT = {:2.3f}sec".format(time.time()-start))

import pickle
with open('fashion_MNIST_SVM.pickle','wb') as f:
    pickle.dump(cvx_solver, f)
# In[ ]:


cvx_solver = pickle.load(open("fashion_MNIST_SVM.pickle", "rb"))


# In[8]:


X_val, val_classes = input_read('fashion_mnist/val.csv', ',')
X_test, test_classes = input_read('fashion_mnist/test.csv', ',')


# In[10]:


label=range(10)
classifiers = list(combinations(label, 2) )
#train_label_score = np.zeros((len(X_train), len(label)))
val_label_score = np.zeros((len(X_val), len(label)))
test_label_score = np.zeros((len(X_test), len(label)))

start = time.time()
for i in range(len(classifiers)):
    k = classifiers[i] # Stores class pair for the current classifier
    X_train, train_classes = input_read('fashion_mnist/train.csv', ',', k[0], k[1])
    #print(k[0], k[1])
    train_classes = train_classes.reshape(-1,1)
    alpha = np.array(cvx_solver[i]['x'])
    S = (alpha > 1e-3)
    supp_vec_ind = np.where(S == True)[0]
    S = S.flatten()
    pdist = spatial.distance.pdist(X_train[supp_vec_ind], 'sqeuclidean')
    K_train = np.exp(-1*0.05*spatial.distance.squareform(pdist))
    w_train = np.dot(K_train.T, (alpha[S]*train_classes[S]))
    bias = train_classes[S] - w_train
    b = np.mean(bias)
    train_pred, val_pred, test_pred= predict(X_train, X_val, X_test, train_classes, val_classes, test_classes, w_train, b, 'gaussian', supp_vec_ind, alpha)    
    #train_pred = [k[1] if x >=0 else k[0] for x in train_pred ]
    
    class1=np.where(val_pred >=0)[0]
    class0=np.where(val_pred <0)[0]
    val_pred[class1] = 1/(1 + np.exp(-np.abs(val_pred[class1])))
    val_pred[class0] = 1/(1 + np.exp(-np.abs(val_pred[class0])))
    val_label_score[class1,k[1]]+=val_pred[class1,0]
    val_label_score[class0,k[0]]+=val_pred[class0,0]
    
    class1=np.where(test_pred >=0)[0]
    class0=np.where(test_pred <0)[0]
    test_pred[class1] = 1/(1 + np.exp(-np.abs(test_pred[class1])))
    test_pred[class0] = 1/(1 + np.exp(-np.abs(test_pred[class0])))
    test_label_score[class1,k[1]]+=test_pred[class1,0]
    test_label_score[class0,k[0]]+=test_pred[class0,0]

print("The time taken to predict the scores is {:2.3f}sec".format(time.time()-start))


# In[11]:


val_label_pred = [np.where(x == max(x))[0][0] for x in val_label_score[:,:]]
test_label_pred  = [np.where(x == max(x))[0][0] for x in test_label_score[:,:]]
val_acc = calc_accuracy(val_classes, val_label_pred)
test_acc = calc_accuracy(test_classes, test_label_pred)
#print(val_acc, test_acc)


# [supp_vec_ind]#Implementation using SVM Classifier

# In[12]:


print("\n-----------GAUSSIAN KERNEL(Multi-class)----------------")
print("---------------CVXOPT------------------------------------")

print("The Validation accuracy of the model with 10C2 classifiers is= {:2.3f}%".format(val_acc))
print("The test accuracy of the model with 10C2 classifiers is = {:2.3f}%".format(test_acc))


# # Read whole input training set for SVM classifier

# In[13]:


X_train, train_classes = input_read('fashion_mnist/train.csv', ',')


# In[14]:


def sklearn_svm(X_train, train_classes, shape, gamma='scale'):
    start = time.time()
    svc_classifier = svc(kernel=shape, gamma=gamma, decision_function_shape='ovo', probability=True)
    svc_classifier.fit(X_train, train_classes)
    train_pred_svc = svc_classifier.predict_proba(X_train)
    val_pred_svc = svc_classifier.predict_proba(X_val)
    test_pred_svc = svc_classifier.predict_proba(X_test)
    train_acc_svc = svc_classifier.score(X_train, train_classes)
    val_acc_svc = svc_classifier.score(X_val, val_classes)
    test_acc_svc = svc_classifier.score(X_test, test_classes)
    print("The time taken to train SVM model using SVM classifier SKLEARN and {} Kernel = {}sec"
          .format(shape, time.time()-start))    
    print("The number of support vectors in SVM classifier trained model with {} kernel = {}"
          .format(shape, svc_classifier.n_support_))
    return train_pred_svc, val_pred_svc, test_pred_svc, train_acc_svc, val_acc_svc, test_acc_svc;


# In[15]:


print("\n\nRunning SVM Classifier from SKLEARN to classify with Gaussian Kernel")
print("--------------------TRAINING--------------------------------------------")
train_pred_svc, val_pred_svc, test_pred_svc, train_acc_svc, val_acc_svc, test_acc_svc= sklearn_svm(X_train, train_classes, 'rbf', 0.05)

val_label_pred_svc = [np.where(x == max(x))[0][0] for x in val_pred_svc[:,:]]
test_label_pred_svc  = [np.where(x == max(x))[0][0] for x in test_pred_svc[:,:]]
val_acc = calc_accuracy(val_classes, val_label_pred_svc)
test_acc = calc_accuracy(test_classes, test_label_pred_svc)

print("-----------SVM classifier--------------")

#print("The number of support vectors are = {}".format())
print("The training accuracy of the SVM model is = {:2.3f}%".format(train_acc_svc*100))
print("The Validation accuracy of the SVM model is= {:2.3f}%".format(val_acc))
print("The test accuracy of the SVM model is = {:2.3f}%".format(test_acc))


# In[21]:


def plot_conf_matrix(kernel_shape, test_classes, test_pred, test_pred_svc ):
    print("\nPlotting Confusion Matrix for {} Kernel - CVXOPT vs. SVMC-library".format(kernel_shape))

    conf_matrix_cvxopt = confusion_matrix(test_classes, test_pred)
    conf_matrix_svmC = confusion_matrix(test_classes, test_pred_svc)
    fig = plt.figure(1)
    ax= fig.add_subplot(111)
    sns.heatmap(conf_matrix_cvxopt, annot=True, ax = ax, fmt="d",linewidths=1, cmap="YlGnBu"); #annot=True to annotate cells
    #ax.set_ylim([0,2]) # Workaround to display values in the center, to avoid downgrade to matplotlib3.1.1
    ax.set_ylabel('Actual labels');
    ax.set_xlabel('Predicted labels'); 
    ax.set_title('Confusion Matrix with CVX '); 
    #ax.xaxis.set_ticklabels(['y=0', 'y=1']); ax.yaxis.set_ticklabels(['y=0', 'y=1']);
    #fig.savefig('cm_cvxopt_test_multiclass.png', dpi=1000, bbox_inches='tight')

    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111)
    sns.heatmap(conf_matrix_svmC, annot=True, ax = ax1, fmt="d",linewidths=1, cmap="YlGnBu"); #annot=True to annotate cells
    #ax1.set_ylim([0,2]) # Workaround to display values in the center, to avoid downgrade to matplotlib3.1.1
    ax1.set_ylabel('Actual labels');
    ax1.set_xlabel('Predicted labels'); 
    ax1.set_title('Confusion Matrix with SVM-C'); 
    #ax1.xaxis.set_ticklabels(['y=0', 'y=1']); ax1.yaxis.set_ticklabels(['y=0', 'y=1']);

    fig.tight_layout(pad=3.0)
    #fig1.savefig('cm_svm_test_multiclass.png', dpi=1000, bbox_inches='tight')
    
    plt.show()


# In[20]:


plot_conf_matrix('gaussian', val_classes, val_label_pred, val_label_pred_svc)


# In[22]:


plot_conf_matrix('gaussian', test_classes, test_label_pred, test_label_pred_svc)


# In[ ]:




