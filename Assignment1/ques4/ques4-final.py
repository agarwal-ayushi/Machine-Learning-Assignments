#!/usr/bin/env python
# coding: utf-8

# In[36]:


#Importing Data from the CSV file
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import math

os.chdir("/home/ayushi/coursework/col774/Machine-Learning-Assignments/Assignment1/data/q4")
print("Path has been changed to:\n", os.getcwd())

#More SOPHISTICATED METHOD FOR DAT FILE
X = np.loadtxt('q4x.dat')
#print(X[1:4])
Y = np.loadtxt('q4y.dat', dtype=str).reshape(-1,1)
#print(Y)

m=X.shape[0]

os.chdir("/home/ayushi/coursework/col774/Machine-Learning-Assignments/Assignment1/ques4")
print("Path changed back to python file directory:\n", os.getcwd())


# In[37]:


#Normalization of the Input X
mean = np.mean(X, axis=0).reshape(1,-1)
var = np.std(X, axis=0).reshape(1,-1)
X-= mean
X/= var
print("Mean of Normalized Data = ",np.mean(X), "\nVariance of Normalized Data=",np.std(X))


# In[38]:


#GDA theta parameters
Y[Y=='Alaska']=1
Y[Y=='Canada']=0
phi = np.sum([Y=='1'])/m

index_x0=[i for i in range(len(Y)) if Y[i]=='0']
index_x1=[i for i in range(len(Y)) if Y[i]=='1']
mean_x0 = (np.mean(X[index_x0], axis=0)).reshape(1,-1)
mean_x1 = (np.mean(X[index_x1], axis=0)).reshape(1,-1)

# Flag = 0 : Function "calculate_cov" returns Covariance of the Data with Labels=0 (Canada)
# Flag = 1 : Function "calculate_cov" returns Covariance of the Data with Labels=1 (Alaska)
# Flag = 0 : Function "calculate_cov" returns Equal Covariance for both labels
def calculate_cov(X, mean_x0, mean_x1, Y, flag):
    cov=np.zeros((2,2))
    cov0=np.zeros((2,2))
    cov1=np.zeros((2,2))
    m = len(X)
    even_count=0; odd_count=0
    for i in range(m):
        if(Y[i]=='1'):
            cov1+=np.dot((X[i:i+1]-mean_x1).T, (X[i:i+1]-mean_x1))
            even_count+=1
        else:
            cov0+=np.dot((X[i:i+1]-mean_x0).T, (X[i:i+1]-mean_x0))
            odd_count+=1
    cov=(cov0+cov1)/m
    cov0/=odd_count
    cov1/=even_count
    if (flag==2): return cov
    if (flag==1): return cov1
    else: return cov0

cov = calculate_cov(X, mean_x0, mean_x1, Y, 2)


# In[39]:


print("Probability PHI for class label='Alaska' is={}".format(phi))
print("Mean of the Distribution for label='Alaska' is={}".format(mean_x1))
print("Mean of the Distribution for label='Canada' is={}".format(mean_x0))
print("The value of the covariance matrix is = \n{}".format(cov))


# In[40]:


#Part (b)
#Plot the training data with class labels
plt.figure(0)
colors=['r' if l=='0' else 'b' for l in Y]
plt.scatter(X[index_x0][:,0], X[index_x0][:,1], s = 5, marker='*', c='r',label='Label=Canada')
plt.scatter(X[index_x1][:,0], X[index_x1][:,1], s = 5, marker='o', c='b',label='Label=Alaska')
plt.legend()
plt.title('(b) Data with labels')
plt.xlabel('Feature -> X1')
plt.ylabel('Feauture -> X2')
plt.show(block=False)
#plt.savefig('data.png', dpi=1000, bbox_inches='tight')


# In[41]:


#calculating linear model parameters
cov_inv = np.linalg.inv(cov)
temp = math.log(phi/(1-phi))
c = (1/2)*((np.dot(np.dot(mean_x1, cov_inv), mean_x1.T)) - (np.dot(np.dot(mean_x0, cov_inv), mean_x0.T)))
m = np.dot(cov_inv, (mean_x1-mean_x0).T)
x_values = np.array([np.min(X[:, 1] -1 ), np.max(X[:, 1] +1 )]).reshape(1,-1)
y = np.dot((-1./m[0:1]),np.dot(m[1:2], x_values)) - c


# In[50]:


#Part (c)
#Plot the training data with class labels
plt.figure(1)
colors=['r' if l=='0' else 'b' for l in Y]
plt.scatter(X[index_x0][:,0], X[index_x0][:,1], s = 5, marker='*', c='r',label='Label=Canada')
plt.scatter(X[index_x1][:,0], X[index_x1][:,1], s = 5, marker='o', c='b',label='Label=Alaska')
plt.title('(c) Data with labels and GDA Decision Boundary')
plt.xlabel('Feature -> X1')
plt.ylabel('Feauture -> X2')
plt.plot(x_values.ravel(), y.ravel(), c='k',label='DB')
plt.legend()
plt.show(block=False)
#plt.savefig('Linear_Decision_Boundary.png', dpi=1000, bbox_inches='tight')


# In[43]:


# visualize Gaussian of the individual distributions

#Create grid and multivariate normal
m1 = np.linspace(-2, 2,500)
m2 = np.linspace(-2, 2,500)
M1, M2 = np.meshgrid(m1,m2)
pos = np.empty(M1.shape + (2,))
pos[:, :, 0] = M1; pos[:, :, 1] = M2
rv1 = multivariate_normal(mean_x1.ravel(), cov)
rv0 = multivariate_normal(mean_x0.ravel(), cov)

#Make a 3D plot
fig = plt.figure(2)
y_ = m[0]*M1 + m[1]*M2 -c

ax = fig.gca(projection='3d')
ax.plot_surface(M1, M2, y_ ,linewidth=0, alpha=0.7)
ax.plot_surface(M1, M2, rv1.pdf(pos),alpha=0.5, cmap='viridis',linewidth=0)
ax.plot_surface(M1, M2, rv0.pdf(pos),alpha=0.5, cmap='viridis', linewidth=0)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Normal Dist')
ax.set_title("(c) Distribution of Data from two classes \nwith linear separator in 3D")
plt.show(block=False)
#plt.savefig('temp.png', dpi=1000, bbox_inches='tight')


# In[44]:


#Create grid and multivariate normal
m1 = np.linspace(-2, 2,500)
m2 = np.linspace(-2, 2,500)
M1, M2 = np.meshgrid(m1,m2)
pos = np.empty(M1.shape + (2,))
pos[:, :, 0] = M1; pos[:, :, 1] = M2
rv1 = multivariate_normal(mean_x1.ravel(), cov)
rv0 = multivariate_normal(mean_x0.ravel(), cov)

#Make a 3D plot
fig = plt.figure(3)
y_ = m[0]*M1 + m[1]*M2 -c

#print(y.shape)
#ax = fig.gca(projection='3d')
ax=fig.add_subplot(111)
ax.contour(M1, M2, y_ , alpha=0.7, cmap='Reds')
ax.contour(M1, M2, rv1.pdf(pos),alpha=0.5, cmap='viridis')
ax.contour(M1, M2, rv0.pdf(pos),alpha=0.5, cmap='viridis')
ax.set_xlabel('X1 feature')
ax.set_ylabel('X2 Feature')
#ax.set_zlabel('Z-axis')
ax.set_title("(c)Contour for showing Linear Decision Boundary")
plt.show(block=False)
#plt.savefig('4c_Contour_DB.png', dpi=1000, bbox_inches='tight')


# In[45]:


# (d) Separate Covariance Matrices. 
cov0 = calculate_cov(X, mean_x0, mean_x1, Y, 0)
cov1 = calculate_cov(X, mean_x0, mean_x1, Y, 1)

print("The value of the covariance matrix for label='Alaska' is = \n{}".format(cov1))
print("The value of the covariance matrix for label='Canada' is = \n{}".format(cov0))
cov0_inv = np.linalg.inv(cov0)
cov1_inv = np.linalg.inv(cov1)

cov0_det = np.linalg.det(cov0)
cov1_det = np.linalg.det(cov1)
temp = math.log(phi/(1-phi))


# In[46]:


a = cov1_inv - cov0_inv
b = 2* (np.dot(cov1_inv, mean_x1.T) - np.dot(cov0_inv,mean_x0.T))
d = np.dot(np.dot(mean_x1, cov1_inv),mean_x1.T) - np.dot(np.dot(mean_x0, cov0_inv),mean_x0.T)- math.log(cov0_det/cov1_det)


# In[47]:


x1_val= np.linspace((np.min(X[:,1])-1), (np.max(X[:,1])+1), 100)
c1 = a[1,1]
x2_val = []
for i in range(len(x1_val)):
    temp2 = (a[0,0]*(x1_val[i]**2)) - (b[0,0]*x1_val[i]) + d[0,0]
    temp3 = ((a[0,1]+a[1,0])*x1_val[i]) - b[1,0]
    x2_val.append(np.roots([c1, temp3, temp2]))

y_q0= [(x2_val[i][0]) for i in range(len(x2_val))]
y_q1= [(x2_val[i][1]) for i in range(len(x2_val))]


# In[49]:


#Part (e)
#Plot the training data with class labels
plt.figure(4)
colors=['r' if l=='0' else 'b' for l in Y]
plt.scatter(X[index_x0][:,0], X[index_x0][:,1], s = 5, marker='*', c='r',label='Label=Canada')
plt.scatter(X[index_x1][:,0], X[index_x1][:,1], s = 5, marker='o', c='b',label='Label=Alaska')
plt.plot(x_values.ravel(), y.ravel(), c='g',label='Linear DB')
plt.title('(e) Data with labels and GDA Linear and Quadratic \n Decision Boundary')
plt.xlabel('Feature -> X1')
plt.ylabel('Feauture -> X2')
plt.scatter(x1_val,y_q0, c='k',s=4,label='Quad DB')
plt.scatter(x1_val,y_q1, c='k',s=4,label='')
plt.ylim(bottom=-4)
plt.ylim(top=6)
plt.legend()
plt.show()
#plt.savefig('Decision_Boundary_Quad.png', dpi=1000, bbox_inches='tight')


# In[ ]:




