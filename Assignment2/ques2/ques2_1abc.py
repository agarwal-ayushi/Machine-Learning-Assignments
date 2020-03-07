#!/usr/bin/env python
# coding: utf-8

# # This is the implementation of the Part B (SVM) . Question 1. a, b, c

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


# 28*28 - one row size = 784 - one training data

# In[2]:


def input_read(file, sep, class0, class1):
    myFile = pd.read_csv(file, sep=sep, header=None, index_col=False)
    df_train = myFile[(myFile.get(784) == class0) | (myFile.get(784) == class1)]
    train_classes = df_train[784]
    train_classes = (train_classes.apply(lambda x: 1 if (x==class1) else -1))*1.
    train_classes.index = np.arange(0, len(df_train))
    df_train = df_train.drop(784, axis=1)
    df_train.index=(np.arange(0, len(df_train)))
    df_train /= 255
    return df_train.to_numpy(), train_classes.to_numpy();


# In[3]:


def linear_svm_cvxopt(X_train, train_classes, c=1):
    sv_neg_ind =[]
    sv_pos_ind=[]
    start = time.time()
    m,n = X_train.shape
    XY = train_classes*X_train
    temp_P= np.dot(XY, XY.T)
    P = cvxopt_matrix(temp_P)
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
    alpha = np.array(cvx_solver['x'])
    S = (alpha > 1e-3)
    supp_vec_ind = np.where(S == True)[0]
    S = S.flatten()
    w2 = np.dot((train_classes[S] * alpha[S]).T, X_train[S]).reshape(-1,1)
    bias = train_classes[S] - np.dot(X_train[S], w2)
    b2 = np.mean(bias)
    print("The time taken to train SVM model using CVXOPT and Linear Kernel = {:2.3f}sec".format(time.time()-start))    
    for i in supp_vec_ind:
        if (train_classes[i] == -1): sv_neg_ind.append(i)
        else: sv_pos_ind.append(i)
    return alpha, w2, b2, supp_vec_ind, sv_neg_ind, sv_pos_ind;


# In[4]:


def calc_accuracy(classes, pred_classes):   
    acc = 0
    for i in range(len(classes)):
        if (classes[i]==pred_classes[i]):
            acc+=1
    acc = (acc/len(classes))*100
    return acc


# In[11]:


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
        train_pred = [1. if x >=0 else -1. for x in train_pred ]
        train_acc = calc_accuracy(train_classes, train_pred)
        
        
        cdist = spatial.distance.cdist(X_train[supp_vec_ind], X_val, 'sqeuclidean')
        K_val = np.exp(-1*0.05*(cdist))
        w = np.dot(K_val.T, (alpha[supp_vec_ind]*train_classes[supp_vec_ind]))
        val_pred = w + b
        val_pred = [1. if x >=0 else -1. for x in val_pred ]
        val_acc = calc_accuracy(val_classes, val_pred)

        cdist = spatial.distance.cdist(X_train[supp_vec_ind], X_test, 'sqeuclidean')
        K_test = np.exp(-1*0.05*(cdist))
        w = np.dot(K_test.T, (alpha[supp_vec_ind]*train_classes[supp_vec_ind]))
        test_pred = w + b
        test_pred = [1 if x >=0 else -1 for x in test_pred ]
        test_acc = calc_accuracy(test_classes, test_pred)
        
    return train_pred, val_pred, test_pred, train_acc, val_acc, test_acc;


# In[5]:


X_train, train_classes = input_read('fashion_mnist/train.csv', ',', 3, 4)
X_val, val_classes = input_read('fashion_mnist/val.csv', ',', 3, 4)
X_test, test_classes = input_read('fashion_mnist/test.csv', ',', 3, 4)


# In[6]:


train_classes = train_classes.reshape(-1,1)
m,n = X_train.shape
c=1 #penalty weight


# In[7]:


print("Running CVX Optimizer to classify Class 3 and Class 4 with Linear Kernel")
print("--------------------TRAINING--------------------------------------------")
alpha, w, b, supp_vec_ind,sv_neg_ind, sv_pos_ind = linear_svm_cvxopt(X_train, train_classes, c)


# In[9]:


train_pred, val_pred, test_pred, train_acc, val_acc, test_acc = predict(X_train, X_val, X_test, train_classes, val_classes, test_classes, w,b, 'linear', supp_vec_ind, alpha)


# In[14]:


print("------------Linear Kernel Results for----------------")
print("----------------------CVXOPT-------------------------\n")
print("The number of support vectors are = {}".format(supp_vec_ind.shape[0]))
print("The number of support vectors per class are = {}".format([len(sv_neg_ind), len(sv_pos_ind)]))
print("The value of b obtained is = {:2.3f}".format(b))
print("The training accuracy of the model is = {:2.3f}%".format(train_acc))
print("The Validation accuracy of the model is= {:2.3f}%".format(val_acc))
print("The test accuracy of the model is = {:2.3f}%".format(test_acc))


# [supp_vec_ind]#Implementation using SVM Classifier

# In[19]:


def sklearn_svm(X_train, train_classes, shape, gamma='scale'):
    start = time.time()
    svc_classifier = svc(kernel=shape, gamma=gamma)
    svc_classifier.fit(X_train, train_classes[:,0])
    train_pred_svc = svc_classifier.predict(X_train)
    val_pred_svc = svc_classifier.predict(X_val)
    test_pred_svc = svc_classifier.predict(X_test)
    train_acc_svc = svc_classifier.score(X_train, train_classes)
    val_acc_svc = svc_classifier.score(X_val, val_classes)
    test_acc_svc = svc_classifier.score(X_test, test_classes)
    print("The time taken to train SVM model using SVM classifier SKLEARN and {} Kernel = {:2.3f}sec"
          .format(shape, time.time()-start))    
    print("The number of support vectors in SVM classifier trained model with {} kernel = {}"
          .format(shape, svc_classifier.n_support_))
    print("The value of b optained from SVMC is = {:2.3f}".format(svc_classifier.intercept_[0]))
    return train_pred_svc, val_pred_svc, test_pred_svc, train_acc_svc, val_acc_svc, test_acc_svc;


# In[21]:


print("\nRunning SVM classifier from SKLEARN to classify Class 3 and Class 4 with Linear Kernel")
print("--------------------TRAINING--------------------------------------------")

print("\n--------------SVM Classifier Results---------------------\n")

train_pred_svc, val_pred_svc, test_pred_svc, train_acc_svc, val_acc_svc, test_acc_svc = sklearn_svm(X_train, train_classes, 'linear')


# In[22]:



print("The training accuracy of the SVM model is = {:2.3f}%".format(train_acc_svc*100))
print("The Validation accuracy of the SVM model is= {:2.3f}%".format(val_acc_svc*100))
print("The test accuracy of the SVM model is = {:2.3f}%".format(test_acc_svc*100))


# In[23]:


def plot_conf_matrix(kernel_shape, test_classes, test_pred, test_pred_svc ):
    print("\nPlotting Confusion Matrix for {} Kernel - CVXOPT vs. SVMC-library".format(kernel_shape))

    conf_matrix_cvxopt = confusion_matrix(test_classes, test_pred)
    conf_matrix_svmC = confusion_matrix(test_classes, test_pred_svc)
    fig = plt.figure()
    ax= fig.add_subplot(221)
    sns.heatmap(conf_matrix_cvxopt, annot=True, ax = ax, fmt="d",linewidths=1, cmap="YlGnBu"); #annot=True to annotate cells
    ax.set_ylim([0,2]) # Workaround to display values in the center, to avoid downgrade to matplotlib3.1.1
    ax.set_ylabel('Actual labels');
    ax.set_xlabel('Predicted labels'); 
    ax.set_title('Confusion Matrix with CVX '); 
    ax.xaxis.set_ticklabels(['y=0', 'y=1']); ax.yaxis.set_ticklabels(['y=0', 'y=1']);

    ax1 = fig.add_subplot(222)
    sns.heatmap(conf_matrix_svmC, annot=True, ax = ax1, fmt="d",linewidths=1, cmap="YlGnBu"); #annot=True to annotate cells
    ax1.set_ylim([0,2]) # Workaround to display values in the center, to avoid downgrade to matplotlib3.1.1
    ax1.set_ylabel('Actual labels');
    ax1.set_xlabel('Predicted labels'); 
    ax1.set_title('Confusion Matrix with SVM-C'); 
    ax1.xaxis.set_ticklabels(['y=0', 'y=1']); ax1.yaxis.set_ticklabels(['y=0', 'y=1']);

    fig.tight_layout(pad=3.0)
    #plt.savefig('cm_gaussian_cvx_svm_partc.png', dpi=1000, bbox_inches='tight')
    
    plt.show()


# In[24]:


#Print confusion matrix (Code help from stackoverflow)

print("------------------Plotting Confusion Matrix for Linear Kernel - CVXOPT vs SVMC--------------------")
plot_conf_matrix('linear', test_classes, test_pred, test_pred_svc)

#Gaussian Kernel
# In[25]:


def gauss_kernel_cvxopt(X_train, train_classes, c=1):
    sv_pos_ind=[]
    sv_neg_ind=[]
    start =time.time()
    m,n = X_train.shape
    K = np.zeros((m,m))
    gamma = 0.05
    start = time.time()
    #     for i in range(m):
    #         for j in range(m):
    #             K[i,j] = np.exp(-1*gamma*np.linalg.norm(X_train[i]-X_train[j])**2)
    pdist = spatial.distance.pdist(X_train, 'sqeuclidean')
    K = np.exp(-1*gamma*spatial.distance.squareform(pdist))
    end = time.time()
    print("Time to Calculate Gaussian Kernel matrix for CVXOPT= {:2.3f}s".format(end-start))

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
    alpha = np.array(cvx_solver['x'])
    S = (alpha > 1e-3)
    supp_vec_ind = np.where(S == True)[0]
    S = S.flatten()
    pdist = spatial.distance.pdist(X_train[supp_vec_ind], 'sqeuclidean')
    K_train = np.exp(-1*0.05*spatial.distance.squareform(pdist))
    w_train = np.dot(K_train.T, (alpha[S]*train_classes[S]))
    bias = train_classes[S] - w_train
    b_train = np.mean(bias)
    print("The time taken to optimize SVM model using CVXOPT and Gaussian Kernel = {:2.3f}sec".format(time.time()-start))    
    for i in supp_vec_ind:
        if (train_classes[i] == -1): sv_neg_ind.append(i)
        else: sv_pos_ind.append(i)
    return alpha, b_train, supp_vec_ind,  sv_neg_ind, sv_pos_ind;


# In[26]:


print("Running CVX Optimizer to classify Class 3 and Class 4 with Gaussian Kernel")
print("--------------------TRAINING--------------------------------------------")
alpha_g, b, supp_vec_ind, sv_neg_ind, sv_pos_ind = gauss_kernel_cvxopt(X_train, train_classes, c)


# In[27]:


train_pred, val_pred, test_pred, train_acc, val_acc, test_acc = predict(X_train, X_val, X_test, train_classes, val_classes, test_classes, w, b, 'gaussian', supp_vec_ind, alpha_g)


# In[28]:


print("\n-----------GAUSSIAN KERNEL----------------")
print("---------------CVXOPT---------------------")

print("The number of support vectors are = {}".format(supp_vec_ind.shape[0]))
print("The number of support vectors per class are = {}".format([len(sv_neg_ind), len(sv_pos_ind)]))
print("The value of b obtained is = {:2.3f}".format(b))
print("The training accuracy of the model is = {:2.3f}%".format(train_acc))
print("The Validation accuracy of the data set is= {:2.3f}%".format(val_acc))
print("The test accuracy of the model is = {:2.3f}%".format(test_acc))


# In[29]:


print("Running SVM Classifier from SKLEARN to classify Class 3 and Class 4 with Gaussian Kernel")
print("\n--------------------TRAINING--------------------------------------------\n")
print("-----------SVM classifier Results--------------")

train_pred_svc, val_pred_svc, test_pred_svc, train_acc_svc, val_acc_svc, test_acc_svc= sklearn_svm(X_train, train_classes, 'rbf', 0.05)


#print("The number of support vectors are = {}".format())
print("The training accuracy of the SVM model is = {:2.3f}%".format(train_acc_svc*100))
print("The Validation accuracy of the SVM model is= {:2.3f}%".format(val_acc_svc*100))
print("The test accuracy of the SVM model is = {:2.3f}%".format(test_acc_svc*100))


# In[30]:


print("\n------------------Plotting Confusion Matrix for Gaussian Kernel - CVXOPT vs SVMC--------------------\n")


plot_conf_matrix('gaussian', test_classes, test_pred, test_pred_svc)


# In[32]:


#Data Plot
print("\n-------------------------Plotting comparison plots--------------------------\n")

data = ["","linear_cvx","linear_svm", "gauss_cvx", "gauss_svm"]
train_acc = [98.64, 98.62, 99.022, 99.02]
val_acc = [93.6, 93.4, 96.4, 96.4]
test_acc = [94.3, 94.3, 96.5, 96.5]
runtime = [65.74,  7.12, 65.45 , 17.77]
sv_num =[493, 493, 1259, 1259]
neg_sv =[259, 259, 627, 627]
pos_sv=[234, 234, 632, 632]

x = range(4)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(x, train_acc, marker='o', label='train accuracy')
ax.plot(x, val_acc, marker='*', label='val accuracy')
ax.plot(x, test_acc, marker='^', label='test accuracy')
ax.set_ylabel('Accuracy in %')
ax.set_xlabel('Optimization technique')
ax.set_title('Accuracy Comparison for \nClassification of\n Class 3 and Class 4')
ax.set_xticklabels(data, rotation = 90)
ax.legend(loc="lower right", fontsize=8)
ax1=fig.add_subplot(122)
ax1.scatter(x, runtime, marker='*', c='r', label='train time')
ax1.legend(loc='center right')
ax1.set_ylabel('Train time (sec)')
ax1.set_xlabel('Optimization technique')
ax1.set_title('Training Time Comparison \nbetween CVXOPT and SVC')
ax1.set_xticklabels(data, rotation = 90)
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
#plt.savefig('accuracy_traintime_comp.png', dpi=1000, bbox_inches='tight')
plt.show()


# In[33]:


fig1=plt.figure()
ax2 = fig1.add_subplot(121)
ax2.scatter(x, neg_sv, marker='*', c='r', label='#Class 3 SV')
ax2.set_ylabel('Number of support vectors')
ax2.set_xlabel('Optimization technique')
ax2.set_title('Comparison of #SV for\n Class 3')
ax2.set_xticklabels(data, rotation = 90)
ax2.legend()
ax3 = fig1.add_subplot(122)
ax3.scatter(x, pos_sv,marker='*', c='r', label='#Class 4 SV')
ax3.set_ylabel('Number of support vectors')
ax3.set_xlabel('Optimization technique')
ax3.set_title('Comparison of #SV for\n Class 4')
ax3.set_xticklabels(data, rotation = 90)
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax3.legend()
#plt.savefig('supp_vec.png', dpi=1000, bbox_inches='tight')

plt.show()


# In[34]:


#variance with number of SV
fig=plt.figure()
ax = fig.add_subplot(111)
sv = [1112, 1238, 1259, 1265, 1312, 1732]
gaus_train_acc = [99.089,99.044 ,99.022,99.067,99.067,99.067]
gaus_val_acc =[96.4,96.4,96.4,96.4,96.4,96.4]
gaus_test_acc =[96.4,96.5,96.5,96.5,96.5,96.5]
ax.plot(sv, gaus_train_acc, marker='o', label='Training Accuracy')
ax.plot(sv, gaus_val_acc, marker='*', label='Validation Accuracy')
ax.plot(sv, gaus_test_acc, marker='^', label='Test Accuracy')
ax.axvline(x=1259, linestyle='--', linewidth=1.2, c='k', label='Optimal')
ax.set_xlabel('Number of support Vectors')
ax.set_ylabel('Accuracy in %')
ax.set_title("Accuracy dependence on #SV of Model trained with \nCVXOPT Gaussian Kernel")
ax.legend()
#plt.savefig('acc_sv_num_gaus.png', dpi=1000, bbox_inches='tight')
plt.show()


# In[35]:


sv = [430, 491, 493, 494, 495, 503, 547]
gaus_train_acc = [15.37, 98.6, 98.64, 98.62, 98.62, 98.62, 98.62]
gaus_val_acc =[14.2, 93.0,93.6,93.6,93.6,93.6,93.6]
gaus_test_acc =[15.4,94.5,94.3,94.3,94.3,94.3,94.3]

fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(sv, gaus_train_acc, marker='o', label='Training Accuracy')
ax.plot(sv, gaus_val_acc, marker='*', label='Validation Accuracy')
ax.plot(sv, gaus_test_acc, marker='^', label='Test Accuracy')
ax.axvline(x=493, linestyle='--', linewidth=1.2, c='k', label='Optimal')
ax.set_ylim(bottom=90)
ax.set_xlabel('Number of support Vectors')
ax.set_ylabel('Accuracy in %')
ax.set_title("Accuracy dependence on #SV of Model trained with \nCVXOPT Linear Kernel")
ax.legend()
#plt.savefig('acc_sv_num_linear.png', dpi=1000, bbox_inches='tight')
plt.show()


# In[ ]:




