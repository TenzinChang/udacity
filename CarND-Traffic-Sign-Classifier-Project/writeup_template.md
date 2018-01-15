# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images-web/30.jpeg "speed limit 30"
[image5]: ./images-web/60.jpeg "speed limit 60"
[image6]: ./images-web/crossing.jpeg "pedestrians"
[image7]: ./images-web/stop.jpeg "stop"
[image8]: ./images-web/thisway.jpeg "keep right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/TenzinChang/udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Label shape = ()
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

i added code to randomly pick a training image and show it. pls refer to the notebook

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I followed the advice on forum and implemented a function: normalize that simply does the following:
```
def normalize(x):
    x = x/255.0 - 0.5
    return x
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is simply reusing LeNet in https://github.com/udacity/CarND-LeNet-Lab - with the following changes:

1. change the input shape from 32x32x1 to 32x32x3 to account for color
2. change output classes from 10 to 43 (#unique traffic signs)
3. add support for dropout. NOTE: dropout is only used during training, not validation/test.

Other than above changes, my layers are exactly identical to the LeNet without any change.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

These are my hyper parameters
EPOCHS = 10
BATCH_SIZE = 128
DROPOUT = 0.5
rate = 0.001

NOTE: i experimented w/ various settings of batch_size, dropout, rate, and finally settled when my validation accuracy
reached 0.94

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.94
* validation set accuracy of 0.94 
* test set accuracy of 0.919

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I simply re-use the LeNet architecture verbatim and only changed the input/output shape. Since this is the first
time I trained on GPU and not sure what the performance is, this was a good start.

* What were some problems with the initial architecture?

1. forgot to normalize validation/test set.
2. add dropout was a bit tricky too - should only dropout during training, not validation/test.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I didn't really change the architecture except adding one dropout layer.

* Which parameters were tuned? How were they adjusted and why?

for network parameters only input/output were changed since orig. LeNet was only handling grayscale 32x32x1, and output 10,
whereas traffic sign is 32x32x3 and output 43.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Adding dropout was a big boost. Also normalization is essential.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The final model perform pretty well. Earlier bugs was because normalization/dropout were applied at the wrong time or to the
wrong dataset, after I fixed them the network perform reached 0.94 with only 10 EPOCHS.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

in the sub folder: images-web:
[image4]: ./images-web/30.jpeg "speed limit 30"
[image5]: ./images-web/60.jpeg "speed limit 60"
[image6]: ./images-web/crossing.jpeg "pedestrians"
[image7]: ./images-web/stop.jpeg "stop"
[image8]: ./images-web/thisway.jpeg "keep right"

My model perform poorly against all these images, only 20% accuracy was achieved. I suspect this is because I didn't normalize
the images the same way as the training data. My image processing skills simply isn't there and even I tried to print out the image before/after normalization I can see what's wrong but don't know why/how to fix them:

- the image AFTER normalization seems to have a lot of noise?
- the colors are all off, after normalization.

I search the forum but can't find any thing obvious. The only tranfromation i used was cv2.resize.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30       		| Speed Limit 30  									| 
| Speed Limit 60      			| No passing 										|
| Pedestrians | General caution |
| Stop					| Priority road											|
| Keep right     		| Keep right			 				|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is way below the test
set accuracy. I suspect I didn't normalize the web images correctly, I visually inspect them but can't seem to fix the color
and noise.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

TopKV2(values=array([[  9.95429158e-01,   4.46533458e-03,   7.32718836e-05,
          1.04713436e-05,   1.01170217e-05],
       [  9.97114658e-01,   9.25948436e-04,   7.79951748e-04,
          5.11433638e-04,   3.50462709e-04],
       [  9.89217758e-01,   5.80937415e-03,   4.93424200e-03,
          2.11232727e-05,   1.75082514e-05],
       [  9.99951243e-01,   3.25301044e-05,   1.61758508e-05,
          5.30919344e-08,   8.67672956e-09],
       [  7.50075698e-01,   2.30126545e-01,   1.49418805e-02,
          3.86013836e-03,   4.63793200e-04]], dtype=float32), indices=array([[ 1,  2,  0,  4, 37],
       [ 9, 16, 12, 13, 28],
       [18,  0,  1, 29, 26],
       [12, 13, 14, 25, 11],
       [34, 30, 38, 23, 11]], dtype=int32))

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


