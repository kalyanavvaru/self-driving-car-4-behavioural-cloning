# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

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

My model consists of a convolution neural network with NvidiaNet architecture. (model.py lines 82-114).
This architecture consists of a series of convolutions, each with 24, 36, 48, 64, 64 filters respectively. 
Every convolution has either 5x5 or 3x3 kernel with 2 pixel stride. At the end of this series of convolutions, there are 4 series of fully connected neural network with hidden layers of 125,50,10, 1 with dropout layers in between with a percentage (~30%) of weights dropped.

The model includes RELU layers to introduce nonlinearity (code line 89-97), and the data is normalized in the model using a Keras lambda layer (code line 85). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 99,105, 109). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 119-126). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate is tuned automatically unlike in SGD based manual learning rate (model.py line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. I have augmented the training data by using below tactics
Created a flipped version of the image and inverse of the angle as additional data (line 50-51). Also added a correction factor of 0.4 to the angle for the images taken from left and right cameras.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict the streeing angle based on the image the camera mounted on the car. 

My first step was to use a convolution neural network model similar to the NvidiaNet architecture. I thought this model might be appropriate because this architecture has proven to be better than LeNet for self driving cars(by Nvidia), also not overtly complicated like GoogLeNet or AlexNet. It has a series of convolutions and stacked on top a series of fully connected layers with dropout layers in between.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Once trained the model I produced was having problems with completing the circuiting. The car veers off track at steep curves. To overcome this I tried a series of experiments modifying various hyper parameters such as batchsize. 

To combat the overfitting, I modified the model so that I introduced dropouts layers. These layers work by dropping weights at random during the neural network training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I tried correcting the angle measurements taken by the left and right cameras included in the training data. The correction factor hyperparameter had positive I tried with 0.1 and 0.2 and 0.3 and ultimately on 0.4 where the car successfully completed the circuit without veering off the lanes.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-114) consisted of a convolution neural network with the following layers and layer sizes

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 36, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 38, 48)         15600     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 18, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 8, 64)          36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 125)               64125     
_________________________________________________________________
dropout_2 (Dropout)          (None, 125)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                6300      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 174,646
Trainable params: 174,646
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I reused the provided training data. The repurposed the training data provided for the counter clockwise lap around the circuit to synthesize also the data for the lap in the reverse direction by flipping image and reversing the angle. I have also corrected the angle for the images represented by cameras on the left and right side of the car.

I then preprocessed this data by normalizing the pixel data by subtracting and diving by 255, the maximum value a pixel can take.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
