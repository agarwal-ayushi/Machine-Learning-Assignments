#!/usr/bin/env python
# coding: utf-8

# In[40]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from xclib.data import data_utils
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import os
import warnings

# In[44]:

warnings.filterwarnings("ignore")
print("----------------Reading the Data-------------------------")
PATH = os.getcwd()
os.chdir('ass3_parta_data/')

X_train = data_utils.read_sparse_file('train_x.txt', force_header=True)
train_class = np.genfromtxt('train_y.txt').reshape(-1, 1)

X_valid = data_utils.read_sparse_file('valid_x.txt', force_header=True)
valid_class = np.genfromtxt('valid_y.txt').reshape(-1, 1)

X_test = data_utils.read_sparse_file('test_x.txt', force_header=True)
test_class = np.genfromtxt('test_y.txt').reshape(-1, 1)


# In[45]:


f = open('train_x.txt') 
m,n = f.readlines()[0].rstrip("\n").split(" ")
m,n = int(m),int(n)
f.close()

print("The number of training samples are = ", m)
print("The number of attributes are = ", n)

os.chdir(PATH)


# ## Experiment with a) n_estimators (50 to 450 in range of 100)

# We will use these parameters - 
# 1. criterion = "entropy"
# 2. max_depth = (default = None) (#this is based on our results)
# 3. min_samples_split - 2 (default)
# 4. mini_samples_leaf - 1 (dafault)
# 5. max_features - auto (default) - Chooses sqrt(n_features)
# 6. bootstrap = True (Default)
# 7. oob_score = False (default) - Whether to use out-of-bag samples to estimate the generalization accuracy.
# 8. n_jobs = 6 
# 9. ccp_alpha = cost complexity pruning - default = 0.0 i.e no pruning is performed
# 

# In[ ]:


#print("-------------------Training different number of RF with different number of estimators----------\n")
#
#
## In[88]:
#
#
#ne_rf = []
#n_estimator = [x for x in range(50, 451, 100)]
#ne_train_acc = []
#ne_val_acc = []
#ne_test_acc = []
#
#for n_esti in n_estimator:
#    ne_rf.append(RandomForestClassifier(n_estimators=n_esti, criterion="entropy", n_jobs=7))
#
#
## In[89]:
#
#
#for i in range(len(ne_rf)):
#    ne_rf[i].fit(X_train.A, train_class.ravel())
#    ne_train_acc.append(ne_rf[i].score(X_train.A, train_class.ravel()))
#    ne_val_acc.append(ne_rf[i].score(X_valid.A, valid_class.ravel()))
#    ne_test_acc.append(ne_rf[i].score(X_test.A, test_class.ravel()))
#
#
## In[90]:
#
#
#ne_train_acc = [x*100 for x in ne_train_acc]
#ne_val_acc = [x*100 for x in ne_val_acc]
#ne_test_acc = [x*100 for x in ne_test_acc]
#
#
## #### Max_depth = Full Trees
#
## In[91]:
#
#
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#
#ax2.plot(n_estimator, ne_val_acc, marker="*", c='b', label="Validation Accuracy")
#ax2.plot(n_estimator, ne_test_acc, marker='^', c='g',label="Test Accuracy")
#ax2.set_xlabel("Number of estimators/Trees in the Forest")
#ax2.set_ylabel("Accuracy in %")
#
#ax1.plot(n_estimator, ne_train_acc, marker='o', c='r',label="Train Accuracy")
#ax1.legend()
#
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.xaxis.tick_top()
#ax1.tick_params(labeltop='off')  # don't put tick labels at the top
#ax2.xaxis.tick_bottom()
#
##This broken axis code taken from matplotlib library
#d = .015  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
#plt.legend()
#
#plt.suptitle("Accuracy of Randon Forest with increasing #Estimators")
#plt.subplots_adjust(hspace=0.05)
##f.savefig("plots/partd/oob/n-estimators-full-depth.png", dpi = 1000)
#plt.show()
#
#
## ## Experiment with b) max_features (0.1 to 1.0 in range of 0.2) - Dropout
## If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
#
## We will use these parameters -
## 
## max_features - use [0.1, 0.3, 0.5, 0.7, 0.9]
## 
## 1. estimators = 100 #Default - gives max accuracy at 350 features
## 2. criterion = "entropy"
## 3. max_depth = None (#this is based on our results)
## 4. min_samples_split - 2 (default)
## 5. mini_samples_leaf - 1 (dafault)
## 6. bootstrap = True (Default)
## 7. oob_score = False (default) - Whether to use out-of-bag samples to estimate the generalization accuracy.
## 8. n_jobs = 6 
## 9. ccp_alpha = cost complexity pruning - default = 0.0 i.e no pruning is performed
## 
#
## In[ ]:
#
#
#print("-------------------Training different number of RF with different number of max features----------\n")
#
#
## In[25]:
#
#
#mf_rf = []
#max_fea = [np.round(x,2) for x in np.arange(0.1, 1.0, 0.2)]
#mf_train_acc = []
#mf_val_acc = []
#mf_test_acc = []
#
#for mf in max_fea:
#    mf_rf.append(RandomForestClassifier(max_features=mf, criterion="entropy", n_jobs=7))
#
#
## In[26]:
#
#
#for i in range(len(mf_rf)):
#    mf_rf[i].fit(X_train.A, train_class.ravel())
#    mf_train_acc.append(mf_rf[i].score(X_train.A, train_class.ravel()))
#    mf_val_acc.append(mf_rf[i].score(X_valid.A, valid_class.ravel()))
#    mf_test_acc.append(mf_rf[i].score(X_test.A, test_class.ravel()))
#
#
## In[139]:
#
#
#max_feature = [x*mf_rf[0].n_features_ for x in max_fea]
#
#
## In[140]:
#
#
#mf_train_acc = [x*100 for x in mf_train_acc]
#mf_val_acc = [x*100 for x in mf_val_acc]
#mf_test_acc = [x*100 for x in mf_test_acc]
#
#
## In[142]:
#
#
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#
#ax2.plot(max_feature, mf_val_acc, marker="*", c='b', label="Validation Accuracy")
#ax2.plot(max_feature, mf_test_acc, marker='^', c='g',label="Test Accuracy")
#
#ax2.set_xlabel("Max-features used to split (fraction of total features)")
#ax2.set_ylabel("Accuracy in %")
#
#ax1.plot(max_feature, mf_train_acc, marker='o', c='r',label="Train Accuracy")
#ax1.legend()
#
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.xaxis.tick_top()
#ax1.tick_params(labeltop='off')  # don't put tick labels at the top
#ax2.xaxis.tick_bottom()
#
#d = .015  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
#plt.legend()
#
#plt.suptitle("Accuracy w.r.t number of features used to split at node")
#plt.subplots_adjust(hspace=0.08)
##f.savefig("plots/partd/oob/max_features-100-withOOB2.png", dpi = 1000, bbox_inches='tight')
#plt.show()
#
#
## #### Max_features exploration Graph with entropy and 450 number of estimators. 
#
## ## Experiment with c) min_sample_split (2 to 10 in range of 2) - Number of leaf
## default - 2
## If int, then consider min_samples_split as the minimum number.
## 
## If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
#
## We will use these parameters -
## 
## min_sample_split = [x for x in range(2, 10, 2)]
## 
## estimators = 450 #Chosen from the previous experiment
## criterion = "entropy"
## max_features = auto (default)
## max_depth = 14 (default = None) (#this is based on our results)
## min_samples_split - 2 (default)
## mini_samples_leaf - 1 (dafault)
## bootstrap = True (Default)
## oob_score = False (default) - Whether to use out-of-bag samples to estimate the generalization accuracy.
## n_jobs = 6 
## ccp_alpha = cost complexity pruning - default = 0.0 i.e no pruning is performed
## 
#
## In[ ]:
#
#
#print("-------------------Training different number of RF with different number of min samples----------\n")
#
#
## In[104]:
#
#
#mss_rf = []
#mss = [x for x in range(2, 11, 2)]
#mss_train_acc = []
#mss_val_acc = []
#mss_test_acc = []
#
#for m in mss:
#    mss_rf.append(RandomForestClassifier(n_estimators=450, min_samples_split=m, criterion="entropy", n_jobs=6))
#
#
## In[105]:
#
#
#for i in range(len(mss_rf)):
#    mss_rf[i].fit(X_train.A, train_class.ravel())
#    mss_train_acc.append(mss_rf[i].score(X_train.A, train_class.ravel()))
#    mss_val_acc.append(mss_rf[i].score(X_valid.A, valid_class.ravel()))
#    mss_test_acc.append(mss_rf[i].score(X_test.A, test_class.ravel()))
#
#
## In[108]:
#
#
#mss_train_acc = [x*100 for x in mss_train_acc]
#mss_val_acc = [x*100 for x in mss_val_acc]
#mss_test_acc = [x*100 for x in mss_test_acc]
#
#
## In[112]:
#
#
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#
#ax2.plot(mss, mss_val_acc, marker="*", c='b', label="Validation Accuracy")
#ax2.plot(mss, mss_test_acc, marker='^', c='g',label="Test Accuracy")
##ax2.plot(mss, mss_rf_oob, marker='+', c='m', label='OOB Score')
#
#ax2.set_xlabel("Minimum #Samples needed to Split")
#ax2.set_ylabel("Accuracy in %")
#
#ax1.plot(mss, mss_train_acc, marker='o', c='r',label="Train Accuracy")
#ax1.legend()
#
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.xaxis.tick_top()
#ax1.tick_params(labeltop='off')  # don't put tick labels at the top
#ax2.xaxis.tick_bottom()
#
#d = .015  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
#plt.legend()
#
#plt.suptitle("Accuracy w.r.t Minimum Samples Needed")
#plt.subplots_adjust(hspace=0.05)
##f.savefig("plots/partd/oob/min_samples.png", dpi = 1000, bbox_inches='tight')
#plt.show()
#
#
## #### min_samples_split exploration Graph with entropy and 450 number of estimators. sqrt(max_features)
#
## ## Experiment with d) RandomState - use integer random seeds either (0 or 42)
## Default  = None - Use global random state from np.random
## 
## Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). 
#
## In[ ]:
#
#
#print("-------------------Training different number of RF with different random state----------\n")
#
#
## In[176]:
#
#
#rand_rf = []
#ran = [0,42] # According to library suggestions
#
#rand_train_acc = []
#rand_val_acc = []
#rand_test_acc = []
#
#
#for r in ran:
#    rand_rf.append(RandomForestClassifier(n_estimators=450, random_state=r, criterion="entropy", n_jobs=7))
#
#
## In[178]:
#
#
#for i in range(len(rand_rf)):
#    rand_rf[i].fit(X_train.A, train_class.ravel())
#    rand_train_acc.append(rand_rf[i].score(X_train.A, train_class.ravel()))
#    rand_val_acc.append(rand_rf[i].score(X_valid.A, valid_class.ravel()))
#    rand_test_acc.append(rand_rf[i].score(X_test.A, test_class.ravel()))
#
#
## In[180]:
#
#
#rand_train_acc = [x*100 for x in rand_train_acc]
#rand_val_acc = [x*100 for x in rand_val_acc]
#rand_test_acc = [x*100 for x in rand_test_acc]
#
#
## #### random state exploration Graph with entropy and 450 number of estimators. sqrt(max_features)
#
## In[186]:
#
#
#fig = plt.figure(1)
#x = [x for x in range(3)]
#lab = [" ","random_state=None", "","","", "random_state=0", "", "", "", "random_state=42"]
#ax1 = fig.add_subplot(111)
#
##ax1.scatter(x, [ne_train_acc[-1], rand_train_acc[0], rand_train_acc[1]], marker="*", c='b', label="Train Accuracy")
#ax1.scatter(x, [ne_val_acc[-1], rand_val_acc[0], rand_val_acc[1]], marker="*", c='r', label="Validation Accuracy")
#ax1.scatter(x, [ne_test_acc[-1], rand_test_acc[0], rand_test_acc[1]], marker="*", c='g', label="Test Accuracy")
#ax1.set_xticklabels(lab, rotation=90)
#ax1.set_ylabel("Accuracy in %")
#plt.title("Accuracy with general random state \n and random state with seed")
#plt.legend()
##plt.savefig("plots/random_state.png", dpi = 1000, bbox_inches='tight')
#plt.show()


# # with Entropy - Parameter search

# ## Using Out of Bag Error and Doing Grid Search
# oob_score bool (default=False)
# 
# Whether to use out-of-bag samples to estimate the generalization accuracy.

# In[ ]:


print("-------------------Performing grid search---------------------------\n")


# In[6]:


from custom_scoring_module import custom_scoring_fuction
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer


# In[7]:


parameters = {'n_estimators':[x for x in range(50,451,100)], 'max_features':[np.round(x,2) for x in np.arange(0.1,1.0,0.2)],
        'min_samples_split':[x for x in range(2,11,2)]}
rfc_search = RandomForestClassifier(criterion='entropy', oob_score=True)


# In[8]:


start =time.time()
# Uncomment this when Grid Search Needed to be run again - It takes 10 hours
# clf = GridSearchCV(rfc_search, parameters, cv=5, scoring=custom_scoring_fuction, n_jobs=7, verbose=3, refit=True)
# clf.fit(X_train.A, train_class.ravel())
end = time.time()


# In[9]:


#print("Time taken to find best estimator={} sec".format(end-start))


# with open("Best_classfier_partc.pickle", 'wb') as f:
#     pickle.dump(clf, f)

# In[14]:
# THE CODE HAS SLIGHTLY BEEN MODIFIED BECAUSE THE GRID SEARCH CANNOT BE DONE WHILE DEMO
# SO WE HAVE SAVED THE FINAL BEST CLASSIFIER AND RETRAINED IT TO GET THE ACCURACY HOWEVER IT IS 
# DIFFERENT FROM WHAT WAS ACHIEVED BY GRID SEARCH REFIT.

best_classifier = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=None, max_features=0.1,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=450, n_jobs=7,
                       oob_score=True, random_state=None, verbose=0,
                       warm_start=False)


# In[47]:


best_classifier.fit(X_train.A, train_class.ravel())


# In[46]:


# with open("Best_classfier_partc.pickle", 'rb') as f:
#     clf = pickle.load(f)

# print("The best Estimator found is = \n", clf.best_estimator_)
# print("\nThe optimal parameters found using OOB error=\n", clf.best_params_)


# In[9]:


#best_classifier = clone(clf.best_estimator_)


# In[48]:


print("The OOB Score of the best estimator is = {:2.4f}".format( best_classifier.oob_score_))
#print(best_classifier.oob_decision_function_)
print("The Training Accuracy of the best estimatot is = {:2.3f}%".format(best_classifier.score(X_train.A, train_class.ravel())*100))
print("The Validation Accuracy of the best estimatot is = {:2.3f}%".format(best_classifier.score(X_valid.A, valid_class.ravel())*100))
print("The Test Accuracy of the best estimatot is = {:2.3f}%".format(best_classifier.score(X_test.A, test_class.ravel())*100))


# # Sensitivity Analysis (Part-D)

# In[ ]:


print("-------------------Sensitivity Analysis of RF with different number of Estimators---------\n")


# #### Sensitivity of the classifier to varying n_estimators (50 to 450 in range of 100) with others at optimum
# 
# max_features=0.1
# 
# min_samples_split = 10

# In[15]:


ne_rf = []
n_estimator = [x for x in range(50, 451, 100)]
ne_train_acc = []
ne_val_acc = []
ne_test_acc = []

for n_esti in n_estimator:
    ne_rf.append(clone(best_classifier).set_params(n_estimators=n_esti))


# In[35]:


for i in range(len(ne_rf)):
    ne_rf[i].fit(X_train.A, train_class.ravel())
    ne_train_acc.append(ne_rf[i].score(X_train.A, train_class.ravel()))
    ne_val_acc.append(ne_rf[i].score(X_valid.A, valid_class.ravel()))
    ne_test_acc.append(ne_rf[i].score(X_test.A, test_class.ravel()))


# In[37]:


ne_train_acc = [x*100 for x in ne_train_acc]
ne_val_acc = [x*100 for x in ne_val_acc]
ne_test_acc = [x*100 for x in ne_test_acc]
ne_rf_oob = [ne_rf[x].oob_score_*100 for x in range(len(ne_rf))]


# In[69]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax2.plot(n_estimator, ne_val_acc, marker="*", c='b', label="Validation Accuracy")
ax2.plot(n_estimator, ne_test_acc, marker='^', c='g',label="Test Accuracy")
ax2.plot(n_estimator, ne_rf_oob, marker='+', c='m', label='OOB Score')

ax2.set_xlabel("Number of estimators/Trees in the Forest")
ax2.set_ylabel("Accuracy in %")

ax1.plot(n_estimator, ne_train_acc, marker='o', c='r',label="Train Accuracy")
ax1.legend()

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

#This broken axis code taken from matplotlib library
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.legend()

plt.suptitle("Sensitivity of Model on #Estimators")
plt.subplots_adjust(hspace=0.05)
#f.savefig("plots/partd/oob/n-estimators-sensitivity.png", dpi = 1000, bbox_inches='tight')
plt.show()


# #### Sensitivity of Classifier to max_features (0.1 to 1.0 in range of 0.2) - Dropout
# 
# n_estimators=150
# 
# min_samples_split=10

# In[ ]:


print("-------------------Sensitivity analysis of RF with different number of Max Features----------\n")


# In[16]:


mf_rf = []
max_fea = [np.round(x,2) for x in np.arange(0.1, 1.0, 0.2)]
mf_train_acc = []
mf_val_acc = []
mf_test_acc = []

for mf in max_fea:
    mf_rf.append(clone(best_classifier).set_params(max_features=mf))


# In[17]:


for i in range(len(mf_rf)):
    mf_rf[i].fit(X_train.A, train_class.ravel())
    mf_train_acc.append(mf_rf[i].score(X_train.A, train_class.ravel()))
    mf_val_acc.append(mf_rf[i].score(X_valid.A, valid_class.ravel()))
    mf_test_acc.append(mf_rf[i].score(X_test.A, test_class.ravel()))


# In[19]:


max_feature = [x*mf_rf[0].n_features_ for x in max_fea]
mf_train_acc = [x*100 for x in mf_train_acc]
mf_val_acc = [x*100 for x in mf_val_acc]
mf_test_acc = [x*100 for x in mf_test_acc]
mf_rf_oob = [mf_rf[x].oob_score_*100 for x in range(len(mf_rf))]


# In[22]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax2.plot(max_feature, mf_val_acc, marker="*", c='b', label="Validation Accuracy")
ax2.plot(max_feature, mf_test_acc, marker='^', c='g',label="Test Accuracy")
ax2.plot(max_feature, mf_rf_oob, marker='+', c='m', label='OOB Score')

ax2.set_xlabel("Max-features used to split (fraction of total features)")
ax2.set_ylabel("Accuracy in %")

ax1.plot(max_feature, mf_train_acc, marker='o', c='r',label="Train Accuracy")
ax1.legend()

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.legend()

plt.suptitle("Sensitivity of Model on max_features")
plt.subplots_adjust(hspace=0.08)
#f.savefig("plots/partd/oob/max_features_senstivity.png", dpi = 1000, bbox_inches='tight')
plt.show()


# #### Sensitivity of the classifier to changing min_sample_split (2 to 10 in range of 2) 
# 
# n_estimators = 150
# 
# max_features=0.1

# In[ ]:


print("-------------------Sensitivity analysis of RF with different number of min samples----------\n")


# In[49]:


mss_rf = []
mss = [x for x in range(2, 11, 2)]
mss_train_acc = []
mss_val_acc = []
mss_test_acc = []

for m in mss:
    mss_rf.append(clone(best_classifier).set_params(min_samples_split=m))


# In[50]:


for i in range(len(mss_rf)):
    mss_rf[i].fit(X_train.A, train_class.ravel())
    mss_train_acc.append(mss_rf[i].score(X_train.A, train_class.ravel()))
    mss_val_acc.append(mss_rf[i].score(X_valid.A, valid_class.ravel()))
    mss_test_acc.append(mss_rf[i].score(X_test.A, test_class.ravel()))


# In[51]:


mss_train_acc = [x*100 for x in mss_train_acc]
mss_val_acc = [x*100 for x in mss_val_acc]
mss_test_acc = [x*100 for x in mss_test_acc]
mss_rf_oob = [mss_rf[x].oob_score_*100 for x in range(len(mss_rf))]


# In[76]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax2.plot(mss, mss_val_acc, marker="*", c='b', label="Validation Accuracy")
ax2.plot(mss, mss_test_acc, marker='^', c='g',label="Test Accuracy")
ax2.plot(mss, mss_rf_oob, marker='+', c='m', label='OOB Score')

ax2.set_xlabel("Minimum #Samples needed to Split")
ax2.set_ylabel("Accuracy in %")

ax1.plot(mss, mss_train_acc, marker='o', c='r',label="Train Accuracy")
ax1.legend()

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.legend()

plt.suptitle("Sensitivity of Model on Minimum Samples Needed")
plt.subplots_adjust(hspace=0.05)
#f.savefig("plots/partd/oob/min_samples_senstivity.png", dpi = 1000, bbox_inches='tight')
plt.show()


# In[ ]:




