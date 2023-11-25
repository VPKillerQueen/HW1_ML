import numpy as np
from util import *
from logistic_np import *
x1_train,y1_train,x1_test,y1_test=get_vehicle_data()
num_train = x1_train.shape[0]
num_test = x1_test.shape[0]
#normalize
x1_train,x1_test=normalize_per_pixel(x1_train,x1_test)
#reshape
x1_train=reshape2D(x1_train)
x1_test=reshape2D(x1_test)
x1_train = x1_train.T
x1_test = x1_test.T
y1_train = y1_train.T
y1_test = y1_test.T
#x2=np.array([[0,1,2],[0,2,3]])
#number of neural
n_h1=10
num_samples=y1_train.shape[1]
#generate weights
W1 = np.random.randn(x1_train.shape[0],n_h1)
b1 = np.random.randn(n_h1,1)
learningrate = 0.01
W4 = np.random.randn(n_h1,y1_train.shape[0])
b4 = np.random.randn(y1_train.shape[0],1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(a):
    return a*(1-a)

def softmax(z):
    z_max = np.array([z.max(axis=1)]).T
    z_quote = np.exp(z-z_max)
    s = np.array([np.sum(z_quote, axis=1)]).T
    softmax = z_quote/s
    return  softmax


type =['Train', 'Test']
for j in type:
    if j == 'Train':
        all_loss = []
        plt.ion()
        for i in range(1000):
            #feedforward
            z1 = W1.T@x1_train + b1
            a1 = sigmoid(z1)
            z4 = W4.T@a1 + b4
            y_hat = sigmoid(z4)

            #backpropagation
            E4 = (y_hat - y1_train)
            D4 = np.multiply(E4, sigmoid_grad(y_hat))
            E1 = W4@D4
            D1 = np.multiply(E1, sigmoid_grad(a1))
            W4 = W4 - learningrate*a1@D4.T
            b4 = b4 - learningrate*np.sum(D4, axis=1, keepdims = True)
            W1 = W1 - learningrate*x1_train@D1.T
            b1 = b1 - learningrate*np.sum(D1, axis=1, keepdims = True)
            cost = -(np.sum(y1_train*np.log(y_hat+10**-6) + (1-y1_train)*np.log(1-y_hat+10**-6)))/num_samples
            all_loss.append(cost)
            if i%10==0:
                plot_loss(all_loss)
                plt.show()
                plt.pause(0.1)
                print("Epoch %d: loss is %.5f" % (i, cost))
    else:
        z1 = W1.T@x1_test + b1
        a1 = sigmoid(z1)
        z4 = W4.T@a1 + b4
        y_hat = sigmoid(z4)
        test(y_hat, y1_test)