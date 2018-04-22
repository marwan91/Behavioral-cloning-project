# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Grayscaling"
[image3]: ./examples/supp1.jpg "Recovery Image"
[image4]: ./examples/supp2.jpg "Recovery Image"
[image5]: ./examples/supp3.jpg "Recovery Image"
[image6]: ./examples/unflipped.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 2 convolution layers neural network with 5x5 filter sizes and depths between 12 and 24 (model.py lines 116-117) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 116-117). 

#### 2. Attempts to reduce overfitting in the model

The model contains  a dropout layer in order to reduce overfitting (model.py lines 121). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 127). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used two different training datasets. The main dataset involved center lane driving across the whole driving track. The second dataset is a small supplementary dataset which focuses on overcoming the weak areas in the main dataset.
Unlike the main dataset , the supplementary dataset is not split into traning and validation sets, rather, it is added to the training dataset.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with the simplest form of convolutional neural networks and adding layer upon layer in such a way to gradually improve the results.

My first step was to normalize the data set to make it easy for the optimizer to tune the weights .
I used a simple convolutional neural network model that consists of 2 convolutional layer. It is a simple architecture but found to be effective.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I experimented with adding dropout layers in various stages in the model architecture and found that adding a dropout layer before the last fully connected layer in the model to improve the validation loss


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I created a supplementary dataset that addresses the weakness in the main dataset, and joined them together.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 131-139) consisted of 3  convolutional layers with 'relu' as the activation function, followed by 3 fully connected layers. 

Anything more than 3 fully connected layers was found worsen the network performance as observed from experimentation, and anything less than 3 convolutional layers was also less effective. So This architecture was the simplest functional one  I could design.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one and a half laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle driving repeatedly through the portions of the track that were found to make the vehicle less likey to remain on the drivable portion of the road.

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Finally , I multiplied all the steering angle values by 3, to make the model more responsive during testing.

After the collection process, I had around 15000 data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the rate at which the validation loss starts to increase ,which is a sign of over fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
