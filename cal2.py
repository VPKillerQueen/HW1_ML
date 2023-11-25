

import numpy as np
from util import *
from softmax_np import *
from logistic_np import *

train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
num_train = train_x.shape[0]
num_val = val_x.shape[0]
num_test = test_x.shape[0]  

train_y = create_one_hot(train_y)
val_y = create_one_hot(val_y)
test_y = create_one_hot(test_y)

#normalize
train_x, val_x, test_x = normalize(train_x, val_x, test_x)
#reshape
train_x = add_one(train_x) 
val_x = add_one(val_x)
test_x = add_one(test_x)
#x2=np.array([[0,1,2],[0,2,3]])
#number of neural
n_h1=10
num_samples=train_y.shape[0]
#generate weights
W1 = np.random.randn(train_x.shape[1],n_h1) #785,10

learningrate = 0.01
W2 = np.random.randn(n_h1,train_y.shape[1]) #10,10


def softmax(x):
    z_max = np.max(x,axis=1,keepdims=True)
    z_quote = np.exp(x-z_max)
    softmax = z_quote/np.sum(z_quote,axis=1,keepdims=True)
    return  softmax

def sigmoid_grad(a):
    return a*(1-a)

def softmax_grad(delta, x): # should be E for delta and input for x, (2500,10) @ (2500,10)
    grad = np.dot(x.T,delta)
    grad = grad/x.shape[0]
    return grad
    
def loss(y,yhat):
    y = y*np.log(yhat)
    return -1/(y.shape[0])*np.sum(np.sum(y,axis = 1)) 

SM = SoftmaxClassifier(10)

type = ['train', 'test']
for i in type:
    if i == 'train':
        for i in range(3500):
            #feedforward
            z1 = train_x@W1#2500,10??
            a1 = z1 #included the feedforward process, this line and the line above, a1 is 2500,10
            z2 = a1@W2  #2500,10
            y_hat = softmax(z2) #included the feedforward process, this line and the line above 2500,10
        #     #backpropagation
            E3 = (y_hat - train_y)
            # D4 = np.multiply(E3, softmax_grad(y_hat)) #2500,10 but in the original file they are 2500,785
            E2 = E3@W2
            # D1 = E2*softmax_grad(E2,train_x)
            W2 = W2 - learningrate*a1.T@E3
            W1 = W1 - learningrate*train_x.T@E2
            # cost = -(np.sum(train_y*np.log(y_hat+10**-6) + (1-train_y)*np.log(1-y_hat+10**-6)))/num_samples
            cost = loss(train_y, y_hat)
            if i%10==0:
                print(i)
                print(cost)
            test1(y_hat,train_y)
    else:
        z1 = test_x@W1#2500,10??
        a1 = softmax(z1) #included the feedforward process, this line and the line above, a1 is 2500,10
        z2 = a1@W2  #2500,10
        y_test_hat = softmax(z2)
        test1(y_test_hat, test_y)
            
