# Behavioural-cloning

## Design

At first, I implemented the model according to the architecture described in the Nvidia paper (the one with input shape (66, 200, 3). The car couldn't start moving at all, as in no instruction coming from the script at all. After reading posts from the forum, I figured out I needed to change the image input size in the drive.py as well, as the resizing is not part of the model. Then I had the second issue that the car would move at constant steering angles from the beginning and drive off the lane. I read through the Slack channel for this project and found out there were some issues with my implementation of the Nvidia model. I modified my model and then the car could drive safely up to the bridge. I proceeded to train the model while including images from the left and the right cameras. After that, the car could move successfully until the second turn after the bridge. I tried different epochs and image augmentation techniques (such as flipping the images) but the car still couldn't get past this point. I thought I could try out a different model and I remembered the CIFAR-10 from the videos. I found the implementation of this model in keras, [CIFAR-10] (https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py), and it enabled the car to drive around the entire track.

## Architecture

As mentioned above, the model is derived from the CIFAR-10 model in keras. and modified the last two layers (replacing Dense(10) and softmax with simply Dense(1) since the the steering angle is just a scalar). There are in total 12 layers including the output layer, out of which four are convolutional layers, two are maxpooling layers, three are drop out layers, and two are fully connected layers (dense). The detailed information of the model is adapted from the output of `model.summary()`.

|Layer (type)                   | Output Shape       |   Param #  |   Connected to|                     
:-----------------------------: |:------------------:| :---------:| :--------------:
convolution2d_1 (Convolution2D) | (None, 32, 32, 32) |   896      |   convolution2d_input_1[0][0]      
convolution2d_2 (Convolution2D) | (None, 30, 30, 32) |   9248     |   convolution2d_1[0][0]            
maxpooling2d_1 (MaxPooling2D)   | (None, 15, 15, 32) |   0        |   convolution2d_2[0][0]            
dropout_1 (Dropout)             | (None, 15, 15, 32) |   0        |   maxpooling2d_1[0][0]             
convolution2d_3 (Convolution2D) | (None, 15, 15, 64) |   18496    |   dropout_1[0][0]                  
convolution2d_4 (Convolution2D) | (None, 13, 13, 64) |   36928    |   convolution2d_3[0][0]            
maxpooling2d_2 (MaxPooling2D)   | (None, 6, 6, 64)   |   0        |   convolution2d_4[0][0]            
dropout_2 (Dropout)             | (None, 6, 6, 64)   |   0        |   maxpooling2d_2[0][0]             
flatten_1 (Flatten)             | (None, 2304)       |   0        |   dropout_2[0][0]                  
dense_1 (Dense)                 | (None, 512)        |   1180160  |   flatten_1[0][0]                  
dropout_3 (Dropout)             | (None, 512)        |   0        |   dense_1[0][0]                    
dense_2 (Dense)                 | (None, 1)          |   513      |   dropout_3[0][0]                  

Total params: 1,246,241

Trainable params: 1,246,241

Non-trainable params: 0


## Process

I ended up using the image data set provided by Udacity. With additionally generated images, there are around 25k images for each epoch. The dataset was split into training and validation sets with a ratio of 8:2. I used Adam as my optimizer with learning rate = 1e-4 rather than the default 1e-3 to prevent overfitting. I compensated the rather small learning rate with 13 epochs of training. I used MSE over cross entropy as the loss function since this is a regression problem rater than a classification one. I applied the generator as suggested, where the width of images were randomly shifted.
