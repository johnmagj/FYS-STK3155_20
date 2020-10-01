# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Where to save the figures and data files

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
    
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
    
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)
    
def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

np.random.seed(3155)

#########################################################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y) #+ 0.1*np.random.randn(20,20)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#########################################################################

# Creating column vector of the z values, turning z (20x20) into an
# array Z (400x1), staring with x_0 for all y's followed
# by x_1 for all y's etc.
Z = np.zeros(np.size(z))
for i in range(len(z)):
        Z[len(z)*i : len(z)*i + len(z)] = z[:,i]
        
# Creating design matrixes as function of polynomials in two variables,
# from degree 1 to 5.
X1 = np.zeros((np.size(z),3))
for i in range(len(x)):
        X1[len(z)*i : len(z)*i + len(z),0] = 1.0
        X1[len(z)*i : len(z)*i + len(z),1] = y[:,i]
        X1[len(z)*i : len(z)*i + len(z),2] = x[:,i]
        
X2 = np.zeros((np.size(z),6))
for i in range(len(x)):
        X2[len(z)*i : len(z)*i + len(z),0] = 1.0
        X2[len(z)*i : len(z)*i + len(z),1] = y[:,i]
        X2[len(z)*i : len(z)*i + len(z),2] = x[:,i]
        X2[len(z)*i : len(z)*i + len(z),3] = x[:,i]*y[:,i]
        X2[len(z)*i : len(z)*i + len(z),4] = y[:,i]**2
        X2[len(z)*i : len(z)*i + len(z),5] = x[:,i]**2
        
X3 = np.zeros((np.size(z),10))
for i in range(len(x)):
        X3[len(z)*i : len(z)*i + len(z),0] = 1.0
        X3[len(z)*i : len(z)*i + len(z),1] = y[:,i]
        X3[len(z)*i : len(z)*i + len(z),2] = x[:,i]
        X3[len(z)*i : len(z)*i + len(z),3] = x[:,i]*y[:,i]
        X3[len(z)*i : len(z)*i + len(z),4] = y[:,i]**2
        X3[len(z)*i : len(z)*i + len(z),5] = x[:,i]**2
        X3[len(z)*i : len(z)*i + len(z),6] = (y[:,i]**2)*x[:,i]
        X3[len(z)*i : len(z)*i + len(z),7] = (x[:,i]**2)*y[:,i]
        X3[len(z)*i : len(z)*i + len(z),8] = y[:,i]**3
        X3[len(z)*i : len(z)*i + len(z),9] = x[:,i]**3
        
X4 = np.zeros((np.size(z),15))
for i in range(len(x)):
        X4[len(z)*i : len(z)*i + len(z),0] = 1.0
        X4[len(z)*i : len(z)*i + len(z),1] = y[:,i]
        X4[len(z)*i : len(z)*i + len(z),2] = x[:,i]
        X4[len(z)*i : len(z)*i + len(z),3] = x[:,i]*y[:,i]
        X4[len(z)*i : len(z)*i + len(z),4] = y[:,i]**2
        X4[len(z)*i : len(z)*i + len(z),5] = x[:,i]**2
        X4[len(z)*i : len(z)*i + len(z),6] = (y[:,i]**2)*x[:,i]
        X4[len(z)*i : len(z)*i + len(z),7] = (x[:,i]**2)*y[:,i]
        X4[len(z)*i : len(z)*i + len(z),8] = y[:,i]**3
        X4[len(z)*i : len(z)*i + len(z),9] = x[:,i]**3
        X4[len(z)*i : len(z)*i + len(z),10] = (x[:,i]**2)*(y[:,i]**2)
        X4[len(z)*i : len(z)*i + len(z),11] = (y[:,i]**3)*x[:,i]
        X4[len(z)*i : len(z)*i + len(z),12] = (x[:,i]**3)*y[:,i]
        X4[len(z)*i : len(z)*i + len(z),13] = y[:,i]**4
        X4[len(z)*i : len(z)*i + len(z),14] = x[:,i]**4
        
X5 = np.zeros((np.size(z),21))
for i in range(len(x)):
        X5[len(z)*i : len(z)*i + len(z),0] = 1.0
        X5[len(z)*i : len(z)*i + len(z),1] = y[:,i]
        X5[len(z)*i : len(z)*i + len(z),2] = x[:,i]
        X5[len(z)*i : len(z)*i + len(z),3] = x[:,i]*y[:,i]
        X5[len(z)*i : len(z)*i + len(z),4] = y[:,i]**2
        X5[len(z)*i : len(z)*i + len(z),5] = x[:,i]**2
        X5[len(z)*i : len(z)*i + len(z),6] = (y[:,i]**2)*x[:,i]
        X5[len(z)*i : len(z)*i + len(z),7] = (x[:,i]**2)*y[:,i]
        X5[len(z)*i : len(z)*i + len(z),8] = y[:,i]**3
        X5[len(z)*i : len(z)*i + len(z),9] = x[:,i]**3
        X5[len(z)*i : len(z)*i + len(z),10] = (x[:,i]**2)*(y[:,i]**2)
        X5[len(z)*i : len(z)*i + len(z),11] = (y[:,i]**3)*x[:,i]
        X5[len(z)*i : len(z)*i + len(z),12] = (x[:,i]**3)*y[:,i]
        X5[len(z)*i : len(z)*i + len(z),13] = y[:,i]**4
        X5[len(z)*i : len(z)*i + len(z),14] = x[:,i]**4 
        X5[len(z)*i : len(z)*i + len(z),15] = (y[:,i]**3)*x[:,i]**2
        X5[len(z)*i : len(z)*i + len(z),16] = (x[:,i]**3)*y[:,i]**2
        X5[len(z)*i : len(z)*i + len(z),17] = (y[:,i]**4)*x[:,i]
        X5[len(z)*i : len(z)*i + len(z),18] = (x[:,i]**4)*y[:,i]
        X5[len(z)*i : len(z)*i + len(z),19] = y[:,i]**5
        X5[len(z)*i : len(z)*i + len(z),20] = x[:,i]**5        
        
#  The design matrix now as function of a given polynomial
X = np.zeros((np.size(z),6))

for i in range(len(x)):
        X[len(z)*i : len(z)*i + len(z),0] = 1.0
        X[len(z)*i : len(z)*i + len(z),1] = y[:,i]
        X[len(z)*i : len(z)*i + len(z),2] = x[:,i]
        X[len(z)*i : len(z)*i + len(z),3] = x[:,i]*y[:,i]
        X[len(z)*i : len(z)*i + len(z),4] = y[:,i]**2
        X[len(z)*i : len(z)*i + len(z),5] = x[:,i]**2
        
def calc(X):
    # We split the data in test and training data
    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)
    # matrix inversion to find beta
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Z_train
    print(beta)
    # and then make the prediction
    Ztilde = X_train @ beta
    print("Training R2")
    print(R2(Z_train,Ztilde))
    print("Training MSE")
    print(MSE(Z_train,Ztilde))
    Zpredict = X_test @ beta
    print("Test R2")
    print(R2(Z_test,Zpredict))
    print("Test MSE")
    print(MSE(Z_test,Zpredict))
    
    return
        
        
# We split the data in test and training data
X1_train, X1_test, Z_train, Z_test = train_test_split(X1, Z, test_size=0.2)
# matrix inversion to find beta
beta = np.linalg.inv(X1_train.T @ X1_train) @ X1_train.T @ Z_train
print(beta)
# and then make the prediction
Ztilde = X1_train @ beta
print("Training R2")
print(R2(Z_train,Ztilde))
print("Training MSE")
print(MSE(Z_train,Ztilde))
Zpredict = X1_test @ beta
print("Test R2")
print(R2(Z_test,Zpredict))
print("Test MSE")
print(MSE(Z_test,Zpredict))

print('##########################################################')

# We split the data in test and training data
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)
# matrix inversion to find beta
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Z_train
print(beta)
# and then make the prediction
Ztilde = X_train @ beta
print("Training R2")
print(R2(Z_train,Ztilde))
print("Training MSE")
print(MSE(Z_train,Ztilde))
Zpredict = X_test @ beta
print("Test R2")
print(R2(Z_test,Zpredict))
print("Test MSE")
print(MSE(Z_test,Zpredict))

print('##########################################################')

# We split the data in test and training data
X5_train, X5_test, Z_train, Z_test = train_test_split(X5, Z, test_size=0.2)
# matrix inversion to find beta
beta = np.linalg.inv(X5_train.T @ X5_train) @ X5_train.T @ Z_train
print(beta)
# and then make the prediction
Ztilde = X5_train @ beta
print("Training R2")
print(R2(Z_train,Ztilde))
print("Training MSE")
print(MSE(Z_train,Ztilde))
Zpredict = X5_test @ beta
print("Test R2")
print(R2(Z_test,Zpredict))
print("Test MSE")
print(MSE(Z_test,Zpredict))