from google.colab import files
file = files.upload()

import matplotlib.pyplot as plt
import random
import time
import numpy as np
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import train_test_split

def diffa(y, ypred,x):
    return (y-ypred)*(-x)

def diffb(y, ypred):
    return (y-ypred)*(-1)

def shuffle_data(x,y):
    # shuffle x，y，while keeping x_i corresponding to y_i
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

def get_batch_data(x, y, batch):
    shuffle_data(x, y)
    x_batch = x[0:batch]
    y_batch = y[0:batch]
    return [x_batch, y_batch]

data = np.loadtxt('LinearRegdata.txt')
x = data[:, 1]
y = data[:, 2]
for i in range(0,5):
  print("x[",i,"] = ",x[i],",","y[",i,"] = ",y[i])

fig = plt.figure();
plt.scatter(x, y)
plt.show()

# Normalize the data
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)
for i in range(0, len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)

# batch gradient descent
def batch_gd(X_train, y_train, X_test, y_test, rate = 0.005, epoch = 150):
    m = 0
    c = 0
    loss_list = []
    ep_list = []

    for i in range(1, epoch+1):
        y_pred = m*X_train + c
        error = y_pred - y_train
        loss = sum(error**2)/(2*len(X_train))

        dm = sum(error * X_train)
        dc = sum(error)
    
        loss_list.append(loss)
        ep_list.append(i)

        m = m - rate*dm
        c = c - rate*dc
        
        y_pred = m*X_test + c
        error = y_pred - y_test
        loss1 = sum(error**2)/2

        if i == 1:
            prev_loss = loss1
            param1 = m
            param2 = c

        else:
            if loss1 < prev_loss:
              prev_loss = loss1
              param1 = m
              param2 = c
            
    fig = plt.figure();
    plt.plot(ep_list, loss_list)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Batch Gradient Descent')
    plt.show()


    fig = plt.figure();
    plt.plot(ep_list, loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Batch Gradient Descent')
    plt.show()
    
    return (param1, param2, prev_loss)

batch_gd(X_train, y_train, X_test, y_test)

def minibatch_gd(X_train, y_train, X_test, y_test, batch_size = 10, rate = 0.02, epochs = 150):
    m = 0
    c = 0
    ep_loss = []
    update_loss = []
    
    for ep in range(epochs):
        loss = 0
        index = np.arange(len(X_train))
        np.random.shuffle(index)
        X = X_train[index]
        y = y_train[index]
        
        for i in np.arange(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            error = m*batch_X + c - batch_y
            loss_ = sum(error**2)/(2* batch_size)
            loss += loss_
            
            dm = sum(error*batch_X)/batch_size
            dc = sum(error)/batch_size
            
            m = m - rate*dm
            c = c - rate*dc
            
            update_loss.append(loss_)
        
        ep_loss.append(loss*batch_size/len(X))
        
        y_pred = m*X_test+c
        error = y_pred - y_test
        loss1 = sum(error**2) / (2*len(X))
        
        if ep == 0:
            prev = loss1
            param1 = m
            param2 = c
        else:
            if prev > loss1:
                prev = loss1
                param1 = m
                param2 = c    
                
    fig = plt.figure();
    plt.plot(np.arange(len(update_loss))+1,update_loss)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title("Mini-Batch Gradient Descent")
    plt.show()


    fig = plt.figure();
    plt.plot(np.arange(len(ep_loss))+1, ep_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Mini-Batch Gradient Descent")
    plt.show()
    
    return (param1, param2, prev)

minibatch_gd(X_train, y_train, X_test, y_test)

def stochastic_gd(X_train, y_train, X_test, y_test, rate = 0.02, epochs = 150):
    m = 0
    c = 0
    ep_loss = []
    update_loss = []
    
    for ep in range(epochs):
        loss = 0
        index = np.arange(len(X_train))
        np.random.shuffle(index)
        X = X_train[index]
        y = y_train[index]
        
        for i in range(len(X)):
            batch_X = X[i]
            batch_y = y[i]
            
            error = m*batch_X + c - batch_y
            loss_ = error**2 / 2 
            loss += loss_
            
            dm = error*batch_X
            dc = error
            
            m = m - rate*dm
            c = c - rate*dc
            
            update_loss.append(loss_)
        
        ep_loss.append(loss/len(X))
        
        y_pred = m*X_test + c
        error = y_pred - y_test
        loss1 = sum(error**2) / (2*len(X))
        
        if ep == 0:
            prev = loss1
            param1 = m
            param2 = c
        else:
            if prev > loss1:
                prev = loss1
                param1 = a
                param2 = b    
                
    fig = plt.figure();
    plt.plot(np.arange(len(update_loss))+1,update_loss)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Stochastic Gradient Descent')
    plt.show()


    fig = plt.figure();
    plt.plot(np.arange(len(ep_loss))+1, ep_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Stochastic Gradient Descent')
    plt.show()
    
    return (param1, param2, prev)

stochastic_gd(X_train, y_train, X_test, y_test)

def momentum_gd(X_train, y_train, X_test, y_test, alpha = 0.02, beta=0.9, epochs = 100):
    m = 0
    c = 0
    vm = 0
    vc = 0
    ep_loss = []
    
    for ep in range(epochs):
        y_pred = m* X_train + c
        error = y_pred - y_train
        loss = sum(error**2)/(2*len(X_train))
        ep_loss.append(loss)
        
        dm = sum(error*X_train)/len(X_train)
        dc = sum(error)/len(X_train)
        
        vm = beta*vm + dm
        vc = beta*vc + dc
        
        m = m - alpha *vm
        c = c - alpha *vc
        
        y_pred = a*X_test + b
        error = y_pred - y_test
        loss1 = sum(error**2)/(2*len(X_test))
        
        if ep == 0:
            prev = loss1
            param1 = m
            param2 = c
        else:
            if prev > loss1:
                previous = loss1
                param1 = m
                param2 = c    
                
    fig = plt.figure();
    plt.plot(np.arange(len(ep_loss))+1,ep_loss)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Momentum Gradient Descent')
    plt.show()


    fig = plt.figure();
    plt.plot(np.arange(len(ep_loss))+1, ep_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Momentum Gradient Descent')
    plt.show()
    
    return (param1, param2, prev)

momentum_gd(X_train, y_train, X_test, y_test)

def adam_gd(X_train, y_train, X_test, y_test, alpha = 0.2, beta1=0.9, beta2=0.999, t = 2, epsilon = 1e-8, epochs = 100):
    m = 0
    c = 0
    vm = 0
    vc = 0
    sm = 0
    sc = 0
    
    ep_loss = []
    
    for ep in range(epochs):
        y_pred = m*X_train + c
        error = y_pred - y_train
        loss = sum(error**2)/(2*len(X_train))
        ep_loss.append(loss)
        
        dm = sum(error*X_train)/len(X_train)
        dc = sum(error)/len(X_train)
        
        vm = beta1*vm + (1-beta1)*dm
        vc = beta1*vc + (1-beta1)*dc
        vm = vm / (1 - beta1**t)
        vc = vc / (1 - beta1**t)
        
        sm = beta2*sm + (1-beta2)*(dm**2)
        sc = beta2*sc + (1-beta2)*(dc**2)
        sm = sm / (1 - beta2**t)
        sc = sc / (1 - beta2**t)
        
        m = m - alpha*vm/(np.sqrt(sm) + epsilon)
        c = c - alpha*vc/(np.sqrt(sc) + epsilon)
        
        y_pred = m*X_test + c
        error = y_pred - y_test
        loss1 = sum(error**2)/(2*len(X_test))
        
        if ep == 0:
            prev = loss1
            param1 = m
            param2 = c

        else:
            if prev > loss1:
                prev = loss1
                param1 = m
                param2 = c    
                
    fig = plt.figure();
    plt.plot(np.arange(len(ep_loss))+1,ep_loss)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Adam Optimizer')
    plt.show()

    
    fig = plt.figure();
    plt.plot(np.arange(len(ep_loss))+1, ep_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Adam Optimizer')
    plt.show()
    
    return (param1,param2,prev)

a,b,prevloss = adam_gd(X_train, y_train, X_test, y_test)
print(a,b,prevloss)

a, b, error1 = batch_gd(X_train, y_train, X_test, y_test)

a, b, error2 = minibatch_gd(X_train, y_train, X_test, y_test)

a, b, error3 = stochastic_gd(X_train, y_train, X_test, y_test)
    
a, b, error4 = momentum_gd(X_train, y_train, X_test, y_test)
    
a, b, error5 = adam_gd(X_train, y_train, X_test, y_test)
    
MSE = [error1, error2, error3, error4, error5]
optimizer = ['batch', 'mini-batch', 'stochastic', 'momentum', 'adam']

RMSE = np.sqrt(MSE)
plt.bar(optimizer, RMSE)
plt.xlabel("Optimizer")
plt.ylabel("RMSE")
plt.show()