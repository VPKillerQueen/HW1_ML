# **MARKDOWN FOR HW1**

## *1. LOGISTCIC REGRESSION WITH NUMPY*

<div style="text-align: justify;">

&nbsp;&nbsp;&nbsp;For the sake of comprehension, the util.py file and the unittest.py will not be explained in this markdown, we will immediately start with logistic_np.py.

&nbsp;&nbsp;&nbsp;The coding file is created based on OOP coding, it defines a class called
**LogisticClassifier** which can be pulled from other code files or self-used in the latter parts of the code. In the class, there are multiple funcitons or definitions which also can be pulled from

### 1.1 `__init__`

&nbsp;&nbsp;&nbsp;The coding file starts with ***innit***, this definition creates multiple variables in order to be used in the same coding file, in other definitions and fuctions. The variable w is initiated using random values following the Xavier/Glorot initialization scheme.

&nbsp;&nbsp;&nbsp;*TODO 1.1: normalize_per_pixel and TODO 1.2: normalize_all_pixel* follow strictly the formular given by the teacher in the *assignment* file which are:

>>>>>>>>>>>>$\overline{x}_{rc}=\frac{a}{b}\sum_{i=0}^{m-1}{x}^{(i)}_{rc}$
>
>>>>>>>>>$\sigma=\sqrt{\frac{1}{m}\sum_{i=0}^{m-1}(x^{{(i)}}_{rc}-\bar{x}_{rc})^{2}}$
>
With $0\le i\le m-1,\ 0\le k\le K-1$ for the process of normalize per pixle and:

>>>>>>>>$\overline{x}=\frac{1}{mRC}\sum_{i=0}^{m-1}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}x_{rc}^{(i)}$
>
>>>>>$\sigma=\sqrt{\frac{1}{mRC}\sum_{i=0}^{m-1}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}\left(x_{rc}^{(i)}-\overline{x}\right)^2}$
>
With $0\le i\le m-1,\ 0\le k\le K-1$ for the processof normalize all pixels.

&nbsp;&nbsp;&nbsp;In the coding file, TODO 1.1:

>> `mean_per_pix = np.sum(train_x, axis = 0 )/train_x.shape[0]`<br>
>>`std_per_pix = np.sqrt(np.sum((x - mean_per_pix)**2 for x in train_x)/train_x.shape[0])`
>
&nbsp;&nbsp;&nbsp;These lines of code calculate the mean and the standard variation per pixel. Therefore, the sum of elements, whose position stayed unchange through out the entire layers of matrixes, are calculated through the mean of summing the matrix. `Axis = 0` means that the sums are calculated along the first axis (axis 0), which corresponds to the index of individual images within the array train_x. In other words, it calculates the sum of pixel values across all training images for each pixel position, effectively computing the mean pixel value per position.

&nbsp;&nbsp;&nbsp;The code calculates the standard deviation per pixel by first computing the squared differences between each pixel value and the mean value for all training images, summing these squared differences, and then dividing by the number of training images (`train_x.shape[0]`). Finally, it takes the square root to obtain the standard deviation per pixel.

`for x in train_x` means that it pull each elements from 2D arrays from each layers
this means that it pulls out matrixes from the first axis, which is the axis 0
the shape of both mean and std are [64,64]

`for i in range(train_x.shape[0]):` <br>
  `train_x[i] = (train_x[i]-mean_per_pix)/std_per_pix` <br>
`for i in range(test_x.shape[0]):`<br>
 `test_x[i] = (test_x[i]-mean_per_pix)/std_per_pix`<br>

The results of this function are demanded to have the shape of [2400,64,64].

This function computes train mean and standard deviation on all pixels then applying data scaling on train_x and test_x using these computed values
:param train_x: train images, shape=(num_train, image_height, image_width)
:param test_x: test images, shape=(num_test, image_height, image_width)

train_mean and train_std should have the shape of (1, image_height, image_width)

`mean = np.full_like(train_x[0],np.mean(train_x))`<br>
`std =  np.full_like(train_x[0],np.std(train_x))`<br>
`for i in range(train_x.shape[0]):
    train_x[i] = (train_x[i]-mean)/std`<br>
`for i in range(test_x.shape[0]):
    test_x[i] = (test_x[i]-mean)/std`<br>

The results here have the similar shape of ones of normalize_per_pix

### 1.2 Reshape

Reshape our 3D tensors to 2D. A 3D tensor of shape (num_samples, image_height, image_width) must be reshaped into (num_samples, image_height*image_width)

`
tensor = tensor.reshape(tensor.shape[0],(tensor.shape[1]*tensor.shape[2]))
`
Output is demanded to have the shape of [number of samples, 64*64], each image is a row vector.

### 1.3 Add one

To calculate dot products easily, we add a column of ones. This can help our model better fit the data and make more accurate predictions.

This function add ones as an additional feature for x

`x = np.concatenate((x,np.ones(x.shape[0]).reshape(-1,1)), axis=1)`

Array of 2400 '1' join in the axis 1
the output of this function has the shape of [5,4097]
which means it has 5 samples with the cojoin features of 64*64 with one additional column of 1

### 1.4 feed_forward

This function compute the output of your logistic classification model, it returns feed forward result (after sigmoid).

$z=x\omega$ <Br>

$\hat{y}=\frac{1}{1+e^{-z}}$ <br>
Compute feedforward result

`z = np.dot(x,self.w)`<br>
`result = 1/(1+np.exp(-z))`<br>

The result here is the predicted output, which is y_hat in the theory and the unit_testpy or this file itself

### 1.5 Compute loss

Compute the loss using y (label) and y_hat (predicted class)

:param y:  the label, the actual class of the samples <br>
:param y_hat: the probability that the given samples belong to class 1 <br>
:return loss: a single value

The formula for calculating the loss is as follows: <br>
$J(\omega)=-\frac{1}{m}\sum_{i=0}^{m-1}\left(y^{(i)}log{\hat{y}}^{(i)}+(1-y^{(i)})log(1-{\hat{y}}^{(i)})\right)$

`
for i in range(y.shape[0]):
>>> loss += y[i]*np.log(y_hat[i])+(1-y[i])*np.log(1-y_hat[i])
loss = np.float64(loss/(-y.shape[0]))
`

This is kinda self-explanatory, the shape is stated pretty clear in this sittuation

### 1.6 Get gradient

Compute and return the gradient of w

$\frac{\partial J(\omega_j)}{\partial\omega_j}=-\frac{1}{m}\sum_{i=0}^{m-1}{({\hat{y}}^{(i)}-y^{(i)})x_j^{(i)}}$

Compute the gradient matrix of `w`, it has the same size of `w`
`
w_grad = np.dot(x.T,(y_hat-y))/y.shape[0]
`

x has the shape of [5,4069] and y has the shape [5,1], the w_grad should have the shape of [1,number of features/weights]

### 1.7 Update weight

Update `w` using the computed gradient: <br>

$\omega=\omega-\alpha\times\frac{\partial J(\omega)}{\partial\omega}$

`
self.w = self.w - learning_rate * grad
`

Since this is a minus operation, the shape of `w` stay the same.

### 1.8 Update weight momentum

Update w using the algorithm with momnetum.

Initialize the momentum matrix before entering the main loop
$∆ω=0$

The `w` update process will take place as follows:

$∆ω=γ∆ω+α\frac{\partial J(\omega)}{\partial\omega}$<br>
$\omega=\omega-∆ω$

As coded:

`momentum = momentum_rate * momentum + learning_rate * grad`<br>
`self.w = self.w - momentum`

### 1.9 Evaluate the classification model

Compute test scores using test_y and y_hat:

`pos_y_hat = np.where(y_hat.round() == 1)`<br>
`pos_test_y = np.where(test_y == 1)`<br>
`pos_neg_test_y = np.where(test_y == 0)`
`TP = len(np.intersect1d(np.array(pos_y_hat),np.array(pos_test_y)))`<br>
`FP = len(np.intersect1d(np.array(pos_y_hat),np.array(pos_neg_test_y)))`<br>

$Precision=\frac{TP}{TP+FP}$

`precision = TP/(TP+FP)`<br>
`recall = TP/(np.array(pos_test_y).shape[1])`<br>
`f1 = 2*precision*recall/(precision+recall)`<br>
`print("Precision: %.3f" % precision)`<br>
`print("Recall: %.3f" % recall)`<br>
`print("F1-score: %.3f" % f1)`<br>
`return precision, recall, f1`<br>

The `np.where()` function is used to find the indices where the condition (for example: `y_hat.round() == 1`) is True. It returns a tuple of arrays, one for each dimension of the input array. In this case, it will return the indices where the condition is satisfied.

</div>

## *2. LOGISTCIC REGRESSION WITH TENSORFLOW*

<div style="text-align: justify;">

TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.

In order to use Tensorflow, one must install Tensorflow on their environment, this should be light work, hence not worth mentioning.

The *logistic_tf.py* coding file uses *util.py* and *logistic_np.py* in the pre-processing and data collecting stages. In particular, it uses the previously mentioned functions *normal_per_pix*, *reshape2D* and *add_one*.

### 2.1 Create TF placeholders to feed train_x and train_y when training

Firstly, we create 2 placeholders for the values of x and y, which is a task given from the *assignment* file. The reason behind these placeholders are the use of empty variables which later on will be feeded in the training process pre-built in the library of Tensorflow and since such library only work with such object, this stage is crucial.

According to the code line:

`x = tf.compat.v1.placeholder(tf.float32, shape = [None,train_x.shape[1]], name = 'input')`<br>
`y = tf.compat.v1.placeholder(tf.float32, shape = [None,train_y.shape[1]], name ='output')`

The placeholders must be created with defined datatype, shape, and name. Here in these code lines, I decided to have them as *float32* since *int* would automatically be rounded and give inaccurate results. The shape should be accorded to the number of features in the input data, which follow the shape of the lastly altered result from *add_one* and the number of samples we decide to use. Therefore, they have the shape of `[None,train_x.shape[1]]`.

### 2.2 Create weights (W) using TF variables

In this step, I created an with tf.variable in the form of a zero filled matrix having the shape `[train_x.shape[1],train_y.shape[1]]`. There are multiple method in creating a new set of weights but due to the powerful implementation of tensorflow itself, a matrix of zeros is sufficient.

`w =  tf.Variable(tf.zeros([train_x.shape[1],train_y.shape[1]]))`

The shape of the weight matrix depends on the next step, which frankly the multiplication of matrix input and weights. Therefore, the  `axis = 0` should fit the `axis = 1` of the input matrix, while the remaining one should resemble the output.

### 2.3 Create a feed-forward operator

Feed forward function start with the multiplication of the altered weight set (or in this case a weight vector) with the input then run the result through an activation function called Sigmoid function. The process is similar to that of the previous section.

In the coding file, this section is done as follow:

`z = tf.matmul(x,w)`<br>
`pred = 1/(1+tf.exp(-z))`

Instead of using `np.dot`, I used `tf.matmul` to ultilize the object in the tensorflow function then the `pred` stay simply a normal calculation.

### 2.4 Write the cost function

The cost function in tensorflow also follow strictly the formular given in `assignment.pdf`, you can see in this code snipet:

`cost = -tf.reduce_sum(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))/num_train`

### 2.5 Create an SGD optimizer and start training

Tensorflow allows user to create a variable that carry the object from the function

`tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)`

This function create an object that act as an optimizer implementing the gradient descent algorithm.

`init = tf.compat.v1.global_variables_initializer()`

This line of code start a Tensorflow object inside a variable called init that initiate nescessary measures for the training.

`with tf.compat.v1.Session() as sess:`

This line of code start a Tensorflow object called sess, which run other mentioned objects of Tensorflow in the same coding file.

`w, loss = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})`

This line of code starts the objects of the optimizer and the calculating of the cost function and updating the result in two variable `w` and `loss`, consecutively.

</div>

## 3. Softmax regression with numpy

<div style="text-align: justify;">

&nbsp;&nbsp;&nbsp;The coding file is created based on OOP coding, it defines a class called
**SoftmaxClassifier** which can be pulled from other code files or self-used in the latter parts of the code. In the class, there are multiple funcitons or definitions which also can be pulled from

### 3.1 Normalize input data

In this file, the technique used to normalize data in this file has been decribed in the first section of this markdown, which is the mean for all pixel method. This code snipet is similar to the one shown previously.

`train_mean = np.mean(train_x)`<br>
`train_std = np.std(train_x)`<br>
`train_x = (train_x - train_mean)/train_std`<br>
`val_x = (val_x - train_mean)/train_std`<br>
`test_x = (test_x - train_mean)/train_std`<br>

### 3.2 Create a one-hot matrix

A one-hot matrix, also known as a one-hot encoding or one-hot vector, is a binary representation of categorical data. It is used to represent categorical variables as binary vectors, where each category is represented by a vector of all zeros except for a single one.

`one_hot_labels = np.zeros((labels.shape[0],num_k))`<br>
`z = 0`<br>
`for i in labels:`<br>
`one_hot_labels[z,i] = 1`<br>
`z += 1`<br>

In this code line, `one_hot_label` has been created by reading the `label` vector, assigning value 1 according to the value in the label vector for each row of the one-hot matrix.

### 3.3 Softmax function and feed forward process

The softmax function follows the formular:

$z=x\omega$,$x\in R^{m \times D}$, $\omega \in R^{D \times K}$

and $z_{max}=\left[max(z^{(0)}),max(z^{(1)}),...,\ max(z^{(m-1)})\right]^T$

$z^{'}= e^{z-z_{max}}$

$s=\ \sum_{k=0}^{K-1}z_k^{\prime\left(i\right)}\ \ ,\ 0\ \le\ i\ \le\ m-1
$

then the feed forward function follows:

${\hat{y}}_k^{(i)}\ =\frac{z_k^{,\left(i\right)}}{8^{\left(i\right)}}\ \ ,\ \ 0\le i\le m-1,\ 0\le k\le K-1$

These softmax function are exercuted in this code snipet:

`z =  np.dot(x,self.w)`<br>
`z_max = np.empty((z.shape[0],1))`<br>
`z_max = np.amax(z, axis = 1)`<br>
`z_max = np.tile(z_max, (10,1)).T`<br>
`z_max = z - z_max`<br>
`r = np.exp(z_max)`<br>

while the feed forward function can be demonstrated as follows:

`x = self.softmax(x)`<br>
`s = np.sum(x,axis=1)`<br>
`for i in range(x.shape[0]):`<br>
`x[i,:] = x[i,:]/s[i]`

### 3.4 Computing loss function

Computing loss function in this scenario is still deviation of the $J(\omega)$, we have:

$J\left(w\right)=\ -\frac{1}{m}\ \sum_{i=0}^{m-1}\sum_{k=0}^{K-1}y_k^{\left(i\right)}log{\hat{y}}_k^{(i)}$

The code snipet demonstrated as:

`for i in range (y.shape[0]):`<br>
`for j in range (y.shape[1]):`<br>
`y[i,j] = np.float32(y[i,j]*np.log(y_hat[i,j]) )`<br>
`return -1/(y.shape[0])*np.sum(np.sum(y,axis = 1))`

### 3.5 Gradient Descent

Gradient descent follow the formular

$\frac{\partial J\left(w\right)}{\partial w}=\frac{1}{m}\ x^T\left(\hat{y}-y\right)$

shown in this snipet code

`grad = np.dot(x.T,(y_hat-y))`<br>
`return grad/x.shape[0]`

### 3.6 Confusion matrix

A confusion matrix is a table used in machine learning and statistics to evaluate the performance of a classification model. It provides a summary of the predicted and actual classifications for a classification problem.

The diagnal line of this matrix is the percentage of True Positive values within a same label (row) while the rest of the values represent the percentage of False Positive.

In the coding file, I used numpy's argmax to gather the position within the `y_hat` and `test_y` to get the index of the highest predicted possibility of the output. This tell me the label each pictures belong to. The data then find the intersection of those array in each label, which mean finding the pictures having the same label. I then take the number of that array, which means the number of pictures having the 'i' label in the predicted array and 'j' label in the actual test array.

It may sound complicated but the code simplify the process:

```
y_hat_label = np.argmax(y_hat,axis=1)
    test_y_label = np.argmax(test_y,axis=1)

    for i in range(10):
        for j in range(10):
            confusion_mat[i,j] = np.intersect1d(np.array(np.where(test_y_label==i)),np.array(np.where(y_hat_label==j))).shape[0]

    confusion_mat = confusion_mat/np.sum(confusion_mat,axis=1)
```

## 4. Softmax with Tensorflow

The softmax with tensorflow work similarly to the logistic tensorlflow, the only difference is the loss, feedforward and cost function.
</div>
