#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.metrics import accuracy
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import activations
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


#  ## Load MNIST dataset


data = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)  = data.load_data()


# ## Normalize the dataset


norm_x_train = x_train/255.0  # or tf.keras.utils.normalize(x_train, axis=1) Among the best practices for training a Neural Network is to normalize your data to obtain a mean close to 0. Normalizing the data generally speeds up learning and leads to faster convergence.
norm_x_test = x_test/255.0  # or tf.keras.utils.normalize(x_test, axis=1)
x_train.shape


# ## Reshape our datasets for filter operation


img_size = 28
norm_x_train = np.array(norm_x_train).reshape(-1,img_size,img_size,1)  # increasing 1 deimension for kernel/filter operation
norm_x_test = np.array(norm_x_test).reshape(-1,img_size,img_size,1)  # increasing 1 deimension for kernel/filter operation
print("New training samples dimension: ", norm_x_train.shape)
print("New test samples dimension: ", norm_x_test.shape)


# 
# # Creating the CNN model


epoch = 15
batches = 64

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3), activation = tf.nn.relu, input_shape = norm_x_train.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(64,(3,3), activation = tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())
          
model.add(tf.keras.layers.Dense(units=64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=32, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units=32, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation = tf.nn.softmax))


# ## Compile our model and train


model.compile(loss = "sparse_categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(norm_x_train, y_train, epochs = epoch, batch_size=batches, validation_split=0.3)


# ## Save the model with .h5  extension for future use. ".h5" stands for Hierarchical data format 5.
# ### Then we load our model for testing.


model.save("my_model.h5")



model =tf. keras.models.load_model('my_model.h5')
model.summary()


# ## History of the model.

hist.history


train_loss = hist.history['loss']
train_acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_accuracy']
xc  = range(epoch)


# ## For the fixed number of epochs we observe the training accuracy vs validation accuracy graph


plt.plot(xc,train_acc, color='green', marker='o', linewidth=2, markerfacecolor='blue')
plt.plot(xc,val_acc, color='magenta',  marker='o', linewidth=2, markerfacecolor='blue')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['Train_Acc','Val_Acc'], loc='lower center')
plt.show()


# ## Evaluate our model on test dataset to check how efficiently our model works


score = model.evaluate(norm_x_test, y_test, verbose=2)
print("Accuracy:", score[1])
print("Loss:", score[0])



# ## Filter Size (3 x 3) and 64 filters are used


layer = model.layers
#print(layer)
filters, biases = model.layers[0].get_weights()
print(layer[0].name, filters.shape)


# ## Plot the filters

col = 8
row = 8
fig1 = plt.figure(figsize=(12,12))
for i in range(1, col*row+1):
    f = filters[:,:,:,i-1]
    fig1 = plt.subplot(row,col,i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:,:,0], cmap = 'gray')
plt.show()

# ## Test on user input or external inputs given


for x in range(0,8):
    img = cv2.imread(f'{x}.png')[:,:, 0 ]
    if(img.shape != (28,28)):
        img = cv2.resize(img,(28,28))
    img = np.invert(np.array([img]))
    img = np.array(img).reshape(-1,img_size,img_size,1)
    predictions = model.predict(img)
    #print(f'{predictions}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.title(f"Prediction: {np.argmax(predictions)}")
    plt.show()
    #predictions.history
    print("Predicted output is:", np.argmax(predictions))




# ## Library needed for number recognition


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ## 2-digits number recognition


img = cv2.imread("50.jpg", cv2.IMREAD_GRAYSCALE)
#gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img)
img.shape


# ## Split the input image into two parts.


x_axs = img.shape[0]//2  # division lines for the picture
y_axs = img.shape[1]//2
print(x_axs)
print(y_axs)

splits0 = img[0:img.shape[0], 0:y_axs]
splits1 = img[0:img.shape[0], y_axs:]



# ## Recognize each part of images seperately.

# ## First part.

plt.imshow(splits0)

if(splits0.shape != (28,28)):
        splits0 = cv2.resize(splits0,(28,28))
print(splits0.shape)
img =np.invert(np.array([splits0]))
img = np.array(img).reshape(-1,img_size,img_size,1)
predictions0 = model.predict(img)


# ## Second part.

plt.imshow(splits1)

if(splits1.shape != (28,28)):
        splits1 = cv2.resize(splits1,(28,28))
print(splits1.shape)

img =np.invert(np.array([splits1]))
img = np.array(img).reshape(-1,img_size,img_size,1)
predictions1 = model.predict(img)


# ## Merge the predicted output of both image parts.

print("Predicted output is:", str(np.argmax(predictions0))+str(np.argmax(predictions1)))


