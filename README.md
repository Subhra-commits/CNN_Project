# CNN_Project
This project describes an approach for efficiently recognizing digits written by different people, taking ingo action different writing styles and ink color. Here we deal with these problems using a multi-layer perceptron image classification problem with the help of the Convolution Neural Network (CNN).
Tensorflow's Keras library and python programming language are used for the entire implementation. 

I have loaded the MNIST dataset for train the model. MNIST dataset is basically consists a set of 70,000 images of handwritten digits from 0 to 9 of 28x28x1 dimension. Among these 70,000 images 60,000 images are used for the training and 10,000 images are used for testing. 

So after loading the MNIST dataset, we split the dataset into training and test set and normalize each dataset. 

Create a sequential model of convolution, pooling and hidden layers. Here I have used the layers in the following order - Conv2d, Pool2d, Conv2d, Pool2d, Dropout, Hidden1, Dropout, Hidden2, Hidden3 and finally the Output layer that determines/predicts the actual output.

*Convolution layer* is used for feature extraction from the image.
*Pooling* actually downsamples our actual data so that training can be speedy. 
*Hidden layer* identifies the important features from input data and use those to correlate between a given input and the correct output. There are no concrete rule of choosing # of hiddn layers. We need to choose by trail and error method. Here I have used three hidden layers. You can use more hidden layers as you need; but keep in mind more number of hidden layers can lead to overfitting. 

*N:B:* Before enter into the hidden layer we must flatten our 2D data into single vector. If we does not flatten our dataset, it will output structured data(spatial). We just want the output of last layer considered as large piece of unstructed data. 

The last layer i.e. output layer produces actual output. 

After Successfully creating our model, we train our model using training dataset for a fixed number of epochs (here 15 epochs are used) and then evaluate on test dataset to check how efficiently our model works/predicts. 

After successful evaluation of our model, we have achieved a validation accuracy of 99.69%. And we test our model for predicting some user given inputs and predict the expected output.
I have tried to extend this program to identify numbers that have more than one digit, here I only done with two digits but in future I want to extend this model for predicting any number of any number of digits. 

That's all about my project.
Thank you. Hope you all like this project. 
