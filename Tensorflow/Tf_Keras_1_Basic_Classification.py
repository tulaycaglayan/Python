# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#Import the Fashion MNIST dataset
#The images are 28x28 NumPy arrays, with pixel values ranging between 0 and 255.
#The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:
#0 T-shirt/top , 1 Trouser ,2 Pullover ,3 Dress ,4 Coat ,5 Sandal ,6 Shirt ,7 Sneaker ,8 Bag ,9 Ankle boot 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Explore the data
#The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
print(train_images.shape)

#Likewise, there are 60,000 labels in the training set
print(len(train_labels))

#Each label is an integer between 0 and 9:
print(train_labels)

#There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
print(test_images.shape)

#And the test set contains 10,000 images labels:
print(len(test_labels))

#Preprocess the data
#The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

#We scale these values to a range of 0 to 1 before feeding to the neural network model.
#For this, cast the datatype of the image components from an integer to a float, and divide by 255. Here's the function to preprocess the images
# TC print(train_images[0][6])
train_images = train_images / 255.0
# TC print(train_images[0][6])

test_images = test_images / 255.0

#Display the first 25 images from the training set and display the class name below each image. Verify that the data is in the correct format and we're ready to build and train the network.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Build the model
#Building the neural network requires configuring the layers of the model, then compiling the model
#Setup the layers
#The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.
#Most of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

print( "model= keras.Sequential completed")

#The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
#Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.

#After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely-connected, or fully-connected, neural layers.
#The first Dense layer has 128 nodes (or neurons). The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
#Each node contains a score that indicates the probability that the current image belongs to one of the 10 digit classes.

#Compile the model
#Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
#• Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
#• Optimizer —This is how the model is updated based on the data it sees and its loss function.
#• Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print( "model.compile completed")

#Train the model
#Training the neural network model requires the following steps:
# 1.Feed the training data to the model—in this example, the train_images and train_labels arrays.
# 2.The model learns to associate images and labels.
# 3.We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array. 

#To start training, call the model.fit method—the model is "fit" to the training data
model.fit(train_images, train_labels, epochs=5)

print( "model.fit completed")

#Evaluate accuracy
#compare how the model performs on the test dataset

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('model.evaluate completed. Test accuracy:', test_acc)
#the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of overfitting.
#Overfitting is when a machine learning model performs worse on new data than on their training data

#Make predictions
#With the model trained, we can use it to make predictions about some images.
predictions = model.predict(test_images)

print("model.predict completed")
      
#Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
print("predictions[0] :" , predictions[0])

#A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing.
#We can see which label has the highest confidence value:

print(" max predictions[0] :" , np.argmax(predictions[0]))

#So the model is most confident that this image is an ankle boot, or class_names[9]. And we can check the test label to see this is correct:
print("test_labels[0] :", test_labels[0])

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
plt.show()
