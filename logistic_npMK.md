# **MARKDOWN FOR HW1**
## *1. LOGISTCIC REGRESSION WITH NUMPY*
<div style="text-align: justify;">
&nbsp;&nbsp;&nbsp;For the sake of comprehension, the util.py file and the unittest.py will not be explained in this markdown, we will immediately start with logistic_np.py.<br>
&nbsp;&nbsp;&nbsp;The coding file is created based on OOP coding, it defines a class called 
<b>LogisticClassifier</b> which can be pulled from other code files or self-used in the latter parts of the code. In the class, there are multiple funcitons or definitions which also can be pulled from

### `__init__`
&nbsp;&nbsp;&nbsp;The coding file starts with <i>__innit__</i>, this definition creates multiple variables in order to be used in the same coding file, in other definitions and fuctions. The variable w is initiated using random values following the Xavier/Glorot initialization scheme.<br>
&nbsp;&nbsp;&nbsp;<i>TODO 1.1: normalize_per_pixel and TODO 1.2: normalize_all_pixel</i> follow strictly the formular given by the teacher in the <i>assignment</i> file which are <br>

>>>>>>>>>>>>$\overline{x}_{rc}=\frac{a}{b}\sum_{i=0}^{m-1}{x}^{(i)}_{rc}$ 
>
>>>>>>>>>$\sigma=\sqrt{\frac{1}{m}\sum_{i=0}^{m-1}(x^{{(i)}}_{rc}-\bar{x}_{rc})^{2}}$
>
With $0\le i\le m-1,\ 0\le k\le K-1$ for the process of normalize per pixle and:<br>

>>>>>>>>$\overline{x}=\frac{1}{mRC}\sum_{i=0}^{m-1}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}x_{rc}^{(i)}$
>
>>>>>$\sigma=\sqrt{\frac{1}{mRC}\sum_{i=0}^{m-1}\sum_{r=0}^{R-1}\sum_{c=0}^{C-1}\left(x_{rc}^{(i)}-\overline{x}\right)^2}$
>
With $0\le i\le m-1,\ 0\le k\le K-1$ for the processof normalize all pixels.<br>
&nbsp;&nbsp;&nbsp;In the coding file, TODO 1.1:<br>

>> `mean_per_pix = np.sum(train_x, axis = 0 )/train_x.shape[0]`<br>
>>`std_per_pix = np.sqrt(np.sum((x - mean_per_pix)**2 for x in train_x)/train_x.shape[0])`
>
&nbsp;&nbsp;&nbsp;These lines of code calculate the mean and the standard variation per pixel. Therefore, the sum of elements, whose position stayed unchange through out the entire layers of matrixes, are calculated through the mean of summing the matrix. `Axis = 0` means that the sums are calculated along the first axis (axis 0), which corresponds to the index of individual images within the array train_x. In other words, it calculates the sum of pixel values across all training images for each pixel position, effectively computing the mean pixel value per position.

&nbsp;&nbsp;&nbsp;The code calculates the standard deviation per pixel by first computing the squared differences between each pixel value and the mean value for all training images, summing these squared differences, and then dividing by the number of training images (`train_x.shape[0]`). Finally, it takes the square root to obtain the standard deviation per pixel.

`for x in train_x` means that it pull each elements from 2D arrays from each layers
this means that it pulls out matrixes from the first axis, which is the axis 0
the shape of both mean and std are [64,64] 

`for i in range(train_x.shape[0]):` <br>
>>    `train_x[i] = (train_x[i]-mean_per_pix)/std_per_pix` <br>
`for i in range(test_x.shape[0]):`<br>
>>    `test_x[i] = (test_x[i]-mean_per_pix)/std_per_pix`<br>

The results of this function are demanded to have the shape of [2400,64,64]
   
This function computes train mean and standard deviation on all pixels then applying data scaling on train_x and test_x using these computed values
:param train_x: train images, shape=(num_train, image_height, image_width)
:param test_x: test images, shape=(num_test, image_height, image_width)

train_mean and train_std should have the shape of (1, image_height, image_width)
```
mean = np.full_like(train_x[0],np.mean(train_x))
std =  np.full_like(train_x[0],np.std(train_x))
for i in range(train_x.shape[0]):
    train_x[i] = (train_x[i]-mean)/std
for i in range(test_x.shape[0]):
    test_x[i] = (test_x[i]-mean)/std
```

The results here have the similar shape of ones of normalize_per_pix

### Reshape
Reshape our 3D tensors to 2D. A 3D tensor of shape (num_samples, image_height, image_width) must be reshaped into (num_samples, image_height*image_width)

```
tensor = tensor.reshape(tensor.shape[0],(tensor.shape[1]*tensor.shape[2]))
```
Output is demanded to have the shape of [number of samples, 64*64], each image is a row vector.

### Add one
To calculate dot products easily, we add a column of ones. This can help our model better fit the data and make more accurate predictions.

This function add ones as an additional feature for x
```
x = np.concatenate((x,np.ones(x.shape[0]).reshape(-1,1)), axis=1)
```
Array of 2400 '1' join in the axis 1
the output of this function has the shape of [5,4097] 
which means it has 5 samples with the cojoin features of 64*64 with one additional column of 1

### feed_forward

This function compute the output of your logistic classification model, it returns feed forward result (after sigmoid).

$z=x\omega$ <Br>
$\hat{y}=\frac{1}{1+e^{-z}}$ <br>
Compute feedforward result
```
z = np.dot(x,self.w)
result = 1/(1+np.exp(-z))
```

The result here is the predicted output, which is y_hat in the theory and the unit_testpy or this file itself

### Compute loss
Compute the loss using y (label) and y_hat (predicted class)

:param y:  the label, the actual class of the samples <br>
:param y_hat: the probability that the given samples belong to class 1 <br>
:return loss: a single value

The formula for calculating the loss is as follows: <br>
$J(\omega)=-\frac{1}{m}\sum_{i=0}^{m-1}\left(y^{(i)}log{\hat{y}}^{(i)}+(1-y^{(i)})log(1-{\hat{y}}^{(i)})\right)$

```
for i in range(y.shape[0]):
>>> loss += y[i]*np.log(y_hat[i])+(1-y[i])*np.log(1-y_hat[i])
loss = np.float64(loss/(-y.shape[0]))
```

This is kinda self-explanatory, the shape is stated pretty clear in this sittuation

### Get gradient
Compute and return the gradient of w

$\frac{\partial J(\omega_j)}{\partial\omega_j}=-\frac{1}{m}\sum_{i=0}^{m-1}{({\hat{y}}^{(i)}-y^{(i)})x_j^{(i)}}$

Compute the gradient matrix of `w`, it has the same size of `w`
```
w_grad = np.dot(x.T,(y_hat-y))/y.shape[0]
```

x has the shape of [5,4069] and y has the shape [5,1], the w_grad should have the shape of [1,number of features/weights]

### Update weight
Update `w` using the computed gradient: <br>

$\omega=\omega-\alpha\times\frac{\partial J(\omega)}{\partial\omega}$

```
self.w = self.w - learning_rate * grad
```

Since this is a minus operation, the shape of `w` stay the same.


### Update weight momentum
Update w using the algorithm with momnetum.

Initialize the momentum matrix before entering the main loop <br>
$∆ω=0$ <br>
The `w` update process will take place as follows:<br>
$∆ω=γ∆ω+α\frac{\partial J(\omega)}{\partial\omega}$ <br>
$\omega=\omega-∆ω$ <br>
As code: 
```
momentum = momentum_rate * momentum + learning_rate * grad
self.w = self.w - momentum
```

### Evaluate the classification model

Compute test scores using test_y and y_hat:
```
pos_y_hat = np.where(y_hat.round() == 1)
pos_test_y = np.where(test_y == 1)
pos_neg_test_y = np.where(test_y == 0)
TP = len(np.intersect1d(np.array(pos_y_hat),np.array(pos_test_y)))
FP = len(np.intersect1d(np.array(pos_y_hat),np.array(pos_neg_test_y)))
```
$Precision=\frac{TP}{TP+FP}$

```
precision = TP/(TP+FP)
recall = TP/(np.array(pos_test_y).shape[1])
f1 = 2*precision*recall/(precision+recall)
print("Precision: %.3f" % precision)
print("Recall: %.3f" % recall)
print("F1-score: %.3f" % f1)
return precision, recall, f1
```
The `np.where()` function is used to find the indices where the condition (for example: `y_hat.round() == 1`) is True. It returns a tuple of arrays, one for each dimension of the input array. In this case, it will return the indices where the condition is satisfied.

</div>