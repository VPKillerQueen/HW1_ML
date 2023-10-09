"""
This file is for multiclass fashion-mnist classification using TensorFlow

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *
# import time
tf.compat.v1.disable_eager_execution() # for some reason, this line of code is required, need further investigation

if __name__ == "__main__":
    np.random.seed(2020)
    tf.compat.v1.set_random_seed(2020)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
   
    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
    x = tf.compat.v1.placeholder(tf.float32, shape = [None,train_x.shape[1]], name = 'input')
    y = tf.compat.v1.placeholder(tf.float32, shape = [None,train_y.shape[1]], name ='output')

    # [TODO 2.8] Create weights (W) using TF variables 
    w =  tf.Variable(tf.zeros([train_x.shape[1],train_y.shape[1]]))

    # [TODO 2.9] Create a feed-forward operator 
    z = tf.matmul(x,w)
    # print(z.shape)
    a = tf.math.reduce_max(z,axis = 1)
    a = tf.reshape(a,(-1,1))
    b = tf.constant([1,10],dtype=tf.int32)
    a = tf.tile(a,b)
    # print(a)
    pred = tf.exp(z-a)
    # print(pred)

    # [TODO 2.10] Write the cost function
    # cost = x.T@(y_hat - y)/x.shape[0]
    cost = -tf.reduce_sum(tf.reduce_sum(y*tf.math.log(pred),axis=1))/num_train

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    # num_epoch = 3347
    learning_rate = 0.01    

    # [TODO 2.8] Create an SGD optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf.compat.v1.global_variables_initializer()
    
    with tf.compat.v1.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            # tic = time.clock()
            # [TODO 2.8] Compute losses and update weights here
            train_loss = sess.run([cost], feed_dict={x:train_x, y:train_y}) 
            val_loss = sess.run([cost], feed_dict={x:val_x, y:val_y}) 
            # Update weights
            sess.run([optimizer], feed_dict={x: train_x, y: train_y})
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            # print(val_loss)
            # toc = time.clock()
            # print(toc-tic)
            # [TODO 2.11] Define your own stopping condition here 
            best_val_loss = float("inf")
            patience = 6  # Number of consecutive increases in validation loss before stopping
            consecutive_increases = 0
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss
                # Optionally, save the model's parameters
                
            else:
                consecutive_increases += 1
            
            if consecutive_increases >= patience:
                print("Stopped due to overfitting.")
                break
        
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)
