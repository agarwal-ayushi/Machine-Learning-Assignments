#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


# In[2]:


print("----------------Reading the Data-------------------------")
PATH = os.getcwd()
os.chdir('Alphabets/')

X_train = pd.read_csv('train.csv', sep=',', header=None, index_col=False)
X_test = pd.read_csv('test.csv', sep=',', header=None, index_col=False)

np.random.shuffle(X_train.to_numpy()) #Randomly shuffle the training data for batching

train_class = X_train[X_train.columns[-1]]
test_actual_class = X_test[X_test.columns[-1]]

X_train = X_train.drop(X_train.columns[-1], axis=1)
X_test = X_test.drop(X_test.columns[-1], axis=1)

print("----------------Data Reading completed-------------------")

os.chdir('../')

print("----------------Preprocessing the data-------------------")
X_train = X_train/255
X_test = X_test/255

m = X_train.shape[0] # Number of Training Samples

#Separating 15% of the training Data as Validation Dataset
X_valid = X_train.iloc[(int(0.85*m)):]
valid_class = train_class[(int(0.85*m)):]
X_train = X_train.iloc[0:int(0.85*m)]
train_class = train_class[0:int(0.85*m)]


m = X_train.shape[0] # Number of Training Samples
n = X_train.shape[1] # Number of input features

print("The total number of training samples = {}".format(m))
print("The total number of validation samples = {}".format(X_valid.shape[0]))
print("The number of features = {}".format(n))


# In[3]:


#To get the one hot encoding of each label
print("--------Perform 1-hot encoding of class labels------------")

train_class_enc = pd.get_dummies(train_class).to_numpy()
valid_class_enc = pd.get_dummies(valid_class).to_numpy()
test_actual_class_enc = pd.get_dummies(test_actual_class).to_numpy()
print("--------------------Done----------------------------------")


# In[4]:


#Add the intercept term to the data samples both in training and test dataset
print("--------Adding the intercept term in the dataset as bias------------")
X_train = np.hstack((np.ones((m,1)),X_train.to_numpy()))
X_valid = np.hstack((np.ones((X_valid.shape[0],1)), X_valid.to_numpy()))
X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test.to_numpy()))
print("-----------------------------Done----------------------------------")


# In[5]:


#Mini-Batch formation

batch_size = 100 # Mini-Batch Size

print("----------------Forming mini-batches of size {}---------------------".format(batch_size))
mini_batch = [(X_train[i:i+batch_size,:], train_class_enc[i:i+batch_size]) for i in range(0, m, batch_size)]
print("The number of mini-batches formed is = {}".format(len(mini_batch)))


# In[19]:


#Theta Initialization 

def theta_init(n, r, arch=[50], mode='normal'):
    theta = []
    for i in range(len(arch)+1):
        if i == 0:
            dim0=n+1
            dim1=arch[i]
        elif (i == len(arch)):
            dim0=arch[i-1]+1
            dim1 = r
        else:
            dim0=arch[i-1]+1
            dim1= arch[i]
        if (mode=='normal'):
            theta.append(np.random.normal(0,0.05, (dim0,dim1)))
        elif(mode=='random'):
            theta.append(2*np.random.random((dim0, dim1))-1)
        elif(mode=='uniform'):
            theta.append(np.random.uniform(-0.05,0.05, (dim0,dim1)))
            
        #theta.append(np.zeros((dim0, dim1)))
    return theta


# In[7]:


# Sigmoid activation function
def activation(x):
    return 1/(1+np.exp(-x))

#ReLU Activation Function
def relu_act(x):
    return np.maximum(0.0, x)

#Derivative of ReLU activation Function
def deriv_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softplus(x):
    return np.log(1+np.exp(x))

def deriv_softplus(x):
    return 1/(1+np.exp(-x))


# In[8]:


# Forward propagation

def forward_prop(data, theta, act_fn='sigmoid'):
    fm = []
    fm.append(data)
    #Sigmoid Activation Function 
    if (act_fn == 'sigmoid'):
        for l in range(len(theta)):
            fm.append(activation(np.dot(fm[l], theta[l])))
            if (l!=len(theta)-1):
                fm[l+1]=np.hstack((np.ones((fm[l+1].shape[0],1)),fm[l+1])) #Add intercept term as bias for each layer
    #ReLU Activation Function 
    elif(act_fn == 'relu'):
        for l in range(len(theta)):
            if (l != len(theta)-1):
                fm.append(relu_act(np.dot(fm[l], theta[l])))
                fm[l+1]=np.hstack((np.ones((fm[l+1].shape[0],1)),fm[l+1])) #Add intercept term as bias for each layer
            else:
                fm.append(activation(np.dot(fm[l], theta[l])))
    #Softplus Activation Function       
    elif(act_fn == 'softplus'):
        for l in range(len(theta)):
            if (l != len(theta)-1):
                fm.append(softplus(np.dot(fm[l], theta[l])))
                fm[l+1]=np.hstack((np.ones((fm[l+1].shape[0],1)),fm[l+1])) #Add intercept term as bias for each layer
            else:
                fm.append(activation(np.dot(fm[l], theta[l])))
    return fm


# In[9]:


# Backward propagation
def backward_prop(fm, Y_b, theta, batch_size, act_fn='sigmoid', cost_fn='sqr_error'):
    delta = [None]*len(fm)
    for l in range(len(fm)-1, 0, -1):
        if (l == len(fm)-1):
            if (cost_fn=='entropy'):
                delta[l] = ((1/batch_size)*((Y_b/fm[l])-((1-Y_b)/(1-fm[l])))*fm[l]*(1-fm[l])) #Entropy Loss
            else:
                delta[l] = ((1/batch_size)*(Y_b - fm[l])*fm[l]*(1-fm[l])) #MSE
        elif (l+1 == len(fm)-1):
            if (act_fn == 'sigmoid'):
                delta[l]=(np.dot(delta[l+1], theta[l].T)*fm[l]*(1-fm[l]))
            elif(act_fn == 'relu'):
                delta[l]=np.dot(delta[l+1], theta[l].T)*deriv_relu(fm[l])
            elif(act_fn=='softplus'):
                delta[l]=np.dot(delta[l+1], theta[l].T)*deriv_softplus(fm[l])
        else:
            if (act_fn == 'sigmoid'):
                delta[l]=(np.dot(delta[l+1][:,1:], theta[l].T)*fm[l]*(1-fm[l])) #To remove delta corresponding to bias node
            elif(act_fn == 'relu'):
                delta[l]=np.dot(delta[l+1][:,1:], theta[l].T)*deriv_relu(fm[l])
            elif(act_fn=='softplus'):
                delta[l]=np.dot(delta[l+1][:,1:], theta[l].T)*deriv_softplus(fm[l])
    return delta


# In[11]:


#Cost Function 
def cost_total(X, theta, Y, m, act_fn='sigmoid', cost_fn='sqr_error'):
    fm = forward_prop(X, theta, act_fn)
    if (cost_fn == 'sqr_error'):
        cost = (1/(2*m))*np.sum((Y-fm[-1])**2) #MSE
    else:
        cost = -(1/m)*(np.sum(((Y*np.log(fm[-1]))+((1-Y)*(np.log(1-fm[-1])))))) #Cross Entropy
    return cost


# In[12]:


def calc_accuracy(data, theta, actual_class, act_fn='sigmoid'):
    pred_class = forward_prop(data, theta, act_fn)
    test_pred_class = pred_class[-1]
    for i in range(len(test_pred_class)):
        test_pred_class[i][test_pred_class[i] == np.max(test_pred_class[i])] = 1 #Choose one with max probability
        test_pred_class[i][test_pred_class[i] != np.max(test_pred_class[i])] = 0


    test_acc = 0
    for i in range(len(actual_class)):
        if (np.array_equal(test_pred_class[i], actual_class[i])):
            test_acc+=1
    test_acc /= data.shape[0]

    return (test_acc*100)


# ## PART AB - One Hidden Layer Neural Network

# In[20]:


def training(mini_batch, X_valid, valid_class_enc, theta, lr, act_fn='sigmoid', lr_mode='constant', cost_fn='sqr_error'):
    lr0=lr
    epoch = 1 # Number of epochs
    early_stop=0 #Early stop count of iteration
    
    cost_init = cost_total(X_valid, theta, valid_class_enc, X_valid.shape[0], act_fn, cost_fn)
    
    while(True):
        count_batch = 0
        #print("Initial Cost on Val dataset for this epoch {} = {}".format(epoch, cost_init))
        
        if(lr_mode == "adaptive"):
            lr = lr0/(np.power(epoch, 1/4))
            if (epoch%10==0):
                print("learning rate for this epoch {} = {:2.4f}".format(epoch, lr))
        
        for b in mini_batch:
            X_b = b[0] 
            Y_b = b[1]
            
            #Forward Propagation
            fm = forward_prop(X_b, theta, act_fn)
            
            #if (count_batch % 60 == 0):
                #print("Error on this batch = "+str(cost_total(X_b, theta, Y_b, batch_size, act_fn, cost_fn)))
                    
            #Backward Propagation
            delta = [None]*len(fm)
            delta = backward_prop(fm, Y_b, theta, batch_size, act_fn, cost_fn)

            #Theta Update
            for t in range(len(theta)):
                if (t == len(theta)-1):
                    theta[t] += lr*np.dot(fm[t].T, delta[t+1]) 
                else:
                    theta[t] += lr*np.dot(fm[t].T, delta[t+1])[:,1:]

            count_batch+=1

        epoch+=1 #Number of epochs
                
        cost_final = cost_total(X_valid, theta, valid_class_enc, X_valid.shape[0], act_fn, cost_fn)
        if (epoch % 10 == 0):
            print("Cost on val dataset after {} epochs is = {:2.5f}".format(epoch, cost_final))
        
        #Stopping criteria for sigmoid - when Validation loss stops decreasing beyond a threshold for 10 epochs
        if (act_fn =='sigmoid'):
            if (abs(cost_final-cost_init) < 1e-05):
                early_stop +=1
            else:
                early_stop=0
            if (early_stop == 10):
                print("cost initial= {:2.5f} , cost final={:2.5f} , change in cost= {:2.5f}".format(cost_init,cost_final, cost_final-cost_init))
                break
        
        #Stopping criteria for relu - when Validation loss increases continuously for 10 epochs
        elif(act_fn=='relu' or act_fn=='softplus'):
            if ((cost_final-cost_init) > 0):
                early_stop +=1
            else:
                early_stop=0
            if (early_stop == 10):
                print("cost initial= {:2.5f} , cost final={:2.5f} , change in cost= {:2.5f}".format(cost_init,cost_final, cost_final-cost_init))
                break
        
        cost_init = cost_final
    return epoch, theta


# In[21]:


def plot_accuracy(arch_test, train_accuracy, test_accuracy, valid_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy with number of hidden units \n in one hidden layer network")
    ax.plot(arch_test, train_accuracy, marker='o', label='Train Accuracy')
    ax.plot(arch_test, valid_accuracy, marker='o',label='Validation Accuracy')
    ax.plot(arch_test, test_accuracy, marker='o', label='Test Accuracy')
    ax.set_xlabel("number of hidden units")
    ax.set_ylabel("Accuracy (%)")

    plt.legend()
    #plt.savefig("plots/partc/accuracy_normal_lr0.5_cube.png", dpi=1000, bbox_inches='tight')
    plt.show()


# In[22]:


def plot_epoch(arch_test, epochs, train_time):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.title("Epochs/Time with number of hidden units \n in one hidden layer network")
    ax.plot(arch_test, epochs, c='b', marker='o', label='#epochs')
    ax.set_xlabel("number of hidden units")
    ax.set_ylabel("Epochs")
    ax.legend()

    ax1 = fig.add_subplot(212)
    ax1.plot(arch_test, train_time, c='b', marker='o', label='train time')
    ax1.set_xlabel("number of hidden units")
    ax1.set_ylabel("train time(sec)")
    plt.legend()
    #plt.savefig("plots/partc/epochs_time_normal_lr0.5_cube.png", dpi=1000, bbox_inches='tight')
    plt.show()


# In[23]:


arch_test = [1,5,10,50,100] # Specifically for part a and b
#arch = [50] #means one hidden layer with 50 perceptrons (DEFAULT)
r = np.max(train_class) + 1 # Default value of the number of classes = 26


# In[65]:

print("-------------------Running Part B with fixed LR---------------------------------\n")

epochs = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
train_time = []

lr=0.5

for i in range(len(arch_test)):
    #Choose between normal or random. Normal gives better results
    theta = theta_init(n, r, [arch_test[i]], 'normal')
    #print(theta[0].shape, theta[1].shape, theta[2].shape)
    print("Training the network with {} hidden layer with {} units".format(len([arch_test[i]]), arch_test[i]))
    print("The parameters of the layers are of the shape:")

    for j in range(len(theta)):
        print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))

    start = time.time()
    epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr, 'sigmoid')
    
    epochs.append(epoch)
    train_time.append(time.time()-start)
    train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc))
    valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc))
    test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc))
    print("\n------------------------------------------------------------------------------")
    print("The stats for number of units in the hidden layer = {} are as below:".format(arch_test[i]))
    print("------------------------------------------------------------------------------")
    print("The number of epochs = {}".format(epochs[-1]))
    print("The training time = {:2.3f}sec".format(train_time[-1]))
    print("The training accuracy is = {:2.3f}%".format(train_accuracy[-1]))
    print("The validation accuracy is = {:2.3f}%".format(valid_accuracy[-1]))
    print("The test accuracy is = {:2.3f}%".format(test_accuracy[-1]))
    print("------------------------------------------------------------------------------\n")


# In[ ]:


print("------------------Plotting Graphs for Part B - Fixed LR - One Hidden Layer ------------------")
plot_accuracy(arch_test, train_accuracy, test_accuracy, valid_accuracy)
plot_epoch(arch_test, epochs, train_time)


# ## Part C - Adaptive Learning rate

# In[90]:

print("-------------------Running Part C with adaptive LR---------------------------------\n")

epochs = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
train_time = []

lr0=1.5

for i in range(len(arch_test)):
    theta = theta_init(n, r, [arch_test[i]], 'normal')
    #print(theta[0].shape, theta[1].shape, theta[2].shape)
    print("Training the network with {} hidden layer with {} units".format(len([arch_test[i]]), arch_test[i]))
    print("The parameters of the layers are of the shape:")

    for j in range(len(theta)):
        print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))

    start = time.time()
    epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr0, 'sigmoid', 'adaptive')
    epochs.append(epoch)
    train_time.append(time.time()-start)
    train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc))
    valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc))
    test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc))
    print("\n------------------------------------------------------------------------------")
    print("The stats for number of units in the hidden layer = {} are as below:".format(arch_test[i]))
    print("------------------------------------------------------------------------------")
    print("The number of epochs = {}".format(epochs[-1]))
    print("The training time = {:2.3f}sec".format(train_time[-1]))
    print("The training accuracy is = {:2.3f}%".format(train_accuracy[-1]))
    print("The validation accuracy is = {:2.3f}%".format(valid_accuracy[-1]))
    print("The test accuracy is = {:2.3f}%".format(test_accuracy[-1]))
    print("------------------------------------------------------------------------------\n")


# In[ ]:


print("------------------Plotting Graphs for Part C - Adaptive LR - One Hidden Layer ------------------")

plot_accuracy(arch_test, train_accuracy, test_accuracy, valid_accuracy)
plot_epoch(arch_test, epochs, train_time)


# ## Part D - Implementation of ReLU activation for Hidden Layer

# In[205]:


print("----------------------------Running Part D-----------------------------------------------------")
print("------------------Training a 100x100 hidden layer network with Sigmoid activation------------------")


# In[112]:


epochs = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
train_time = []


# In[235]:


arch=[100, 100]
lr=1.5
theta = theta_init(n, r, arch, 'normal')
print("Training the network with {} hidden layer with {} units".format(len(arch), arch))
print("The parameters of the layers are of the shape:")

for j in range(len(theta)):
    print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))
    
start = time.time()
epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr, 'sigmoid', 'adaptive')

epochs.append(epoch)
train_time.append(time.time()-start)
train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc))
valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc))
test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc))

print("\n------------------------------------------------------------------------------")
print("The stats for number of units in the hidden layer = {} with Sigmoid are as below:".format(arch))
print("------------------------------------------------------------------------------")
print("The number of epochs with Sigmoid is = {}".format(epochs[-1]))
print("The training time with Sigmoid is = {:2.3f}sec".format(train_time[-1]))
print("The training accuracy with Sigmoid is = {:2.3f}%".format(train_accuracy[-1]))
print("The validation accuracy with Sigmoid is = {:2.3f}%".format(valid_accuracy[-1]))
print("The test accuracy with Sigmoid is = {:2.3f}%".format(test_accuracy[-1]))
print("------------------------------------------------------------------------------\n")


# In[93]:


print("----------------------------Running Part D-----------------------------------------------------")
print("------------------Training a 100x100 hidden layer network with ReLU activation------------------")


# In[96]:


arch=[100, 100]
lr=0.6
theta = theta_init(n, r, arch, 'uniform')
print("Training the network with {} hidden layer with {} units".format(len(arch), arch))
print("The parameters of the layers are of the shape:")

for j in range(len(theta)):
    print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))
    
start = time.time()
epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr, 'relu', 'adaptive')

epochs.append(epoch)
train_time.append(time.time()-start)
train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc, 'relu'))
valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc, 'relu'))
test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc, 'relu'))

print("\n------------------------------------------------------------------------------")
print("The stats for number of units in the hidden layer = {} with ReLU are as below:".format(arch))
print("------------------------------------------------------------------------------")
print("The number of epochs with ReLU is = {}".format(epochs[-1]))
print("The training time with ReLU is = {:2.3f}sec".format(train_time[-1]))
print("The training accuracy with ReLU is = {:2.3f}%".format(train_accuracy[-1]))
print("The validation accuracy with ReLU is = {:2.3f}%".format(valid_accuracy[-1]))
print("The test accuracy with ReLU is = {:2.3f}%".format(test_accuracy[-1]))
print("------------------------------------------------------------------------------\n")


# In[116]:


print("----------------------------Running Part D-----------------------------------------------------")
print("------------------Training a 100x100 hidden layer network with SoftPlus activation------------------")


# In[234]:


arch=[100, 100]
lr=0.6
theta = theta_init(n, r, arch, 'normal')
print("Training the network with {} hidden layer with {} units".format(len(arch), arch))
print("The parameters of the layers are of the shape:")

for j in range(len(theta)):
    print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))
    
start = time.time()
epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr, 'softplus', 'adaptive')

epochs.append(epoch)
train_time.append(time.time()-start)
train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc, 'softplus'))
valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc, 'softplus'))
test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc, 'softplus'))

print("\n------------------------------------------------------------------------------")
print("The stats for number of units in the hidden layer = {} with softplus are as below:".format(arch))
print("------------------------------------------------------------------------------")
print("The number of epochs with softplus is = {}".format(epochs[-1]))
print("The training time with softplus is = {:2.3f}sec".format(train_time[-1]))
print("The training accuracy with softplus is = {:2.3f}%".format(train_accuracy[-1]))
print("The validation accuracy with softplus is = {:2.3f}%".format(valid_accuracy[-1]))
print("The test accuracy with softplus is = {:2.3f}%".format(test_accuracy[-1]))
print("------------------------------------------------------------------------------\n")


# In[157]:


print("------------------Plotting Graphs for Part D - ReLU of 2 Hidden Layer ARCH ------------------")
x=[0,1,2]
data = [' ','ReLU', ' ' ,' ' ,' ','Sigmoid', ' ',' ',' ','Softplus']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Accuracy and Epochs for Neural Network [100,100]")
ax.plot(x, train_accuracy, marker='o', label='Train Accuracy')
ax.plot(x, valid_accuracy, marker='o',label='Validation Accuracy')
ax.plot(x, test_accuracy, marker='o', label='Test Accuracy')
ax.set_xlabel("Activation Function")
ax.set_xticklabels(data)
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax1=ax.twinx()
ax1.set_ylabel("Number of Epochs")
ax1.plot(x, epochs, marker='*', color='m',label='Epochs')
ax1.tick_params(axis='y', labelcolor='m', labelsize=12)
ax1.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig("plots/partd/comp_diff_act.png", dpi=1000, bbox_inches='tight')
plt.show()


# ### Part F - Binary Cross Entropy With ReLU activation

# In[178]:


epochs=[]
train_accuracy = []
valid_accuracy = []
test_accuracy = []
train_time = []


# In[179]:


print("----------------------------Running Part F-----------------------------------------------------")
print("------------------Training a 100x100 hidden layer network with ReLU activation and Cross Entropy------------------")


# In[180]:


arch=[100, 100]
lr=0.1
theta = theta_init(n, r, arch, 'normal')
print("Training the network with {} hidden layer with {} units".format(len(arch), arch))
print("The parameters of the layers are of the shape:")

for j in range(len(theta)):
    print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))
    
start = time.time()
epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr, 'relu', 'adaptive', 'entropy')

epochs.append(epoch)
train_time.append(time.time()-start)
train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc, 'relu'))
valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc, 'relu'))
test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc, 'relu'))

print("\n------------------------------------------------------------------------------")
print("The stats for number of units in the hidden layer = {} with ReLU are as below:".format(arch))
print("------------------------------------------------------------------------------")
print("The number of epochs with ReLU is = {}".format(epochs[-1]))
print("The training time with ReLU is = {:2.3f}sec".format(train_time[-1]))
print("The training accuracy with ReLU is = {:2.3f}%".format(train_accuracy[-1]))
print("The validation accuracy with ReLU is = {:2.3f}%".format(valid_accuracy[-1]))
print("The test accuracy with ReLU is = {:2.3f}%".format(test_accuracy[-1]))
print("------------------------------------------------------------------------------\n")


# In[181]:


print("----------------------------Running Part F-----------------------------------------------------")
print("------------------Training a 100x100 hidden layer network with Softplus activation and Cross Entropy------------------")


# In[182]:


arch=[100, 100]
lr=0.1
theta = theta_init(n, r, arch, 'normal')
print("Training the network with {} hidden layer with {} units".format(len(arch), arch))
print("The parameters of the layers are of the shape:")

for j in range(len(theta)):
    print("theta between layer {} and layer {} is {}".format(j, j+1,theta[j].shape))
    
start = time.time()
epoch, theta = training(mini_batch, X_valid, valid_class_enc, theta, lr, 'softplus', 'adaptive', 'entropy')

epochs.append(epoch)
train_time.append(time.time()-start)
train_accuracy.append(calc_accuracy(X_train, theta, train_class_enc, 'softplus'))
valid_accuracy.append(calc_accuracy(X_valid, theta, valid_class_enc, 'softplus'))
test_accuracy.append(calc_accuracy(X_test, theta, test_actual_class_enc, 'softplus'))

print("\n------------------------------------------------------------------------------")
print("The stats for number of units in the hidden layer = {} with softplus are as below:".format(arch))
print("------------------------------------------------------------------------------")
print("The number of epochs with softplus is = {}".format(epochs[-1]))
print("The training time with softplus is = {:2.3f}sec".format(train_time[-1]))
print("The training accuracy with softplus is = {:2.3f}%".format(train_accuracy[-1]))
print("The validation accuracy with softplus is = {:2.3f}%".format(valid_accuracy[-1]))
print("The test accuracy with softplus is = {:2.3f}%".format(test_accuracy[-1]))
print("------------------------------------------------------------------------------\n")


# In[189]:


print("------------------Plotting Graphs for Part F - ReLU with Cross Entropy ------------------")
x=[0,1]
data = [' ','ReLU', ' ' ,' ' ,' ',' ','Softplus']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Accuracy and Epochs for Neural Network [100,100]\n with Cross-Entropy")
ax.plot(x, train_accuracy, marker='o', label='Train Accuracy')
ax.plot(x, valid_accuracy, marker='o',label='Validation Accuracy')
ax.plot(x, test_accuracy, marker='o', label='Test Accuracy')
ax.set_xlabel("Activation Function")
ax.set_xticklabels(data)
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax1=ax.twinx()
ax1.set_ylabel("Number of Epochs")
ax1.plot(x, epochs, marker='*', color='m',label='Epochs')
ax1.tick_params(axis='y', labelcolor='m', labelsize=12)
plt.legend(loc=4)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig("plots/partf/comp_diff_act.png", dpi=1000, bbox_inches='tight')
plt.show()


# ## Part E - MLP Classifier using SKLEARN

# In[238]:


print("----------------------------Running Part E-----------------------------------------------------")
print("------------------Training a 100x100 hidden layer network with MLP Classifier------------------")


# In[98]:


from sklearn.neural_network import MLPClassifier as mlp_classifier
PATH = os.getcwd()
os.chdir('Alphabets/')
X_train = pd.read_csv('train.csv', sep=',', header=None, index_col=False)
X_test = pd.read_csv('test.csv', sep=',', header=None, index_col=False)
train_class = X_train[X_train.columns[-1]]
test_actual_class = X_test[X_test.columns[-1]]

X_train = X_train.drop(X_train.columns[-1], axis=1)
X_test = X_test.drop(X_test.columns[-1], axis=1)

os.chdir('../')

X_train = X_train/255
X_test = X_test/255

m = X_train.shape[0] # Number of Training Samples
n = X_train.shape[1] # Number of input features
train_class_enc = pd.get_dummies(train_class).to_numpy()
test_actual_class_enc = pd.get_dummies(test_actual_class).to_numpy()


# In[123]:


epochs = []
train_accuracy = []
test_accuracy = []
train_time = []


# In[124]:


clf=[]
#Classifier - logistic with constant LR
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='constant', max_iter=400,
                     tol=1e-5, verbose=False))
#Classifier - logistic with early_stopping =False
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='constant', max_iter=400,
                     early_stopping=False, tol=1e-5, verbose=False))
#Classifier - logistic with invscaling with lr0=0.3 and p=5
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', 
                     batch_size=100, learning_rate_init=0.3, learning_rate='invscaling', max_iter=400,
                     power_t=(1/5), tol=1e-5, verbose=False))
#Classifier - logistic with invscaling with pow(1/3)
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', 
                     batch_size=100, learning_rate_init=0.3, learning_rate='invscaling', max_iter=400,
                     power_t=(1/4), tol=1e-5, verbose=False))


#Classifier ReLU with constant LR (1e-5) lr=0.03
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.03, learning_rate='constant', max_iter=400,
                     tol=1e-5, verbose=False))
#Classifier ReLU with constant LR (1e-5) lr=0.1
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='constant', max_iter=400,
                     tol=1e-5, verbose=False))
#Classifier ReLU with constant LR (1e-5) with early stopping
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='constant', max_iter=400,
                     early_stopping=False, tol=1e-5, verbose=False))

#Classifier ReLU with constant LR (1e-6) with invscaling sqrt
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='invscaling', max_iter=400,
                     power_t=(1/2), tol=1e-5, verbose=False))
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='invscaling', max_iter=400,
                     power_t=(1/3), tol=1e-5, verbose=False))
#Classifier ReLU with constant LR (1e-6) with invscaling pow(1/3)
clf.append(mlp_classifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', 
                     batch_size=100, learning_rate_init=0.1, learning_rate='invscaling', max_iter=400,
                     power_t=(1/4), tol=1e-5, verbose=False))


# In[125]:


for i in range(len(clf)):
    print("--------Training the classifier with following params--------------")
    print("-------------------------------------------------------------------")
    print(clf[i])
    print("-------------------------------------------------------------------")
    start =time.time()
    clf[i].fit(X_train, train_class_enc)
    end = time.time()
    epochs.append(clf[i].n_iter_)
    train_accuracy.append(clf[i].score(X_train, train_class_enc)*100)
    test_accuracy.append(clf[i].score(X_test, test_actual_class_enc)*100)
    train_time.append(end-start)
    print("The training Accuracy achieved is = {:2.3f}".format(train_accuracy[-1]))
    print("The test Accuracy achieved is = {:2.3f}".format(test_accuracy[-1]))    
    print("The number of epochs is = {}".format(epochs[-1]))    
    print("The training time achieved is = {:2.3f}".format(train_time[-1]))    


# In[149]:


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
ax.set_title("Accuracy with different MLP classifiers with\n ReLU and Sigmoid Activation")
data=['S_lr=0.1', 'S_earlystop','S_0.3/ep^(1/5)','S_0.3/ep^(1/4)','R_lr=0.03', 'R_lr=0.1', 'R_EarlyStop', 'R_0.1/ep^(1/2)', 'R_0.1/ep^(1/3)', 'R_0.1/ep^(1/4)']
x=np.arange(len(clf))
ax.plot(x, train_accuracy, marker='o', label='Train Accuracy')
ax.plot(x, test_accuracy, marker='o', label='Test Accuracy')
ax.set_xlabel("Classifier Type")
ax.set_xticklabels(data, rotation=90)
ax.set_ylabel("Accuracy (%)")
ax.legend(framealpha=0.5)
ax.set_xticks(range(0,len(clf)))
ax1=ax.twinx()
ax1.set_ylabel("Number of Epochs")
ax1.plot(x, epochs, marker='*', color='m',label='Epochs')
ax1.tick_params(axis='y', labelcolor='m', labelsize=12)
plt.legend(loc=4,framealpha=0.5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig("plots/parte/accuracy_final.png", dpi=1000, bbox_inches='tight')
plt.show()


# In[ ]:




