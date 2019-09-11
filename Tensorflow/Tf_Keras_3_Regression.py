import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

#In a regression problem, we aim to predict the output of a continuous value, like a price or a probability.
#Contrast this with a classification problem, where we aim to predict a discrete label (for example, where a picture contains an apple or an orange).
#This notebook builds a model to predict the median price of homes in a Boston suburb during the mid-1970s. To do this, we'll provide the model with
#some data points about the suburb, such as the crime rate and the local property tax rate.

#The Boston Housing Prices dataset
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print(type(train_data), type(train_labels),type(test_data), type(test_labels))

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Examples and features
#This dataset is much smaller than the others we've worked with so far: it has 506 total examples are split between 404 training examples and 102 test examples
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

#The dataset contains 13 different features
#Each one of these input data features is stored using a different scale. Some features are represented by a proportion between 0 and 1,
#other features are ranges between 1 and 12, some are ranges between 0 and 100, and so on. This is often the case with real-world data,
#and understanding how to explore and clean such data is an important skill to develop.

#Key Point: As a modeler and developer, think about how this data is used and the potential benefits and harm a model's predictions can cause.
#A model like this could reinforce societal biases and disparities. Is a feature relevant to the problem you want to solve or will it introduce bias? For more information, read about ML fairness.
 
print(train_data[0])  # Display sample features, notice the different scales

import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(" Train data features : ")
print(df.head())

#Labels
#The labels are the house prices in thousands of dollars. (You may notice the mid-1970s prices.)
print(" Train data  prices : " , train_labels[0:10])  # Display first 10 entries

df = pd.DataFrame(test_data, columns=column_names)
print(" Test data features : ")
print(df.head())
print(" Test data prices : ", test_labels[0:10])  # Display first 10 entries


#Normalize features
#It's recommended to normalize features that use different scales and ranges. For each feature, subtract the mean of the feature and divide by the standard deviation
# Test data is *not* used when calculating the mean and std.

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print("Normalize features  completed " , train_data[0])  # First training sample, normalized

#Although the model might converge without feature normalization, it makes training more difficult, and it makes the resulting model more dependant on the choice of units used in the input.

#Create the model
#Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers, and an output layer that returns a single, continuous value.
#The model building steps are wrapped in a function, build_model, since we'll create a second model, later on.

print("train_data.shape[1] : ", train_data.shape[1])

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()
print("Create the model completed ") 

#Train the model
#The model is trained for 500 epochs, and record the training and validation accuracy in the history object.

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])

print("Train the model completed")

#Visualize the model's training progress using the stats stored in the history object. We want to use this data to determine how long to train before the model stops making progress
import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,5])
  plt.show()

plot_history(history)

#This graph shows little improvement in the model after about 200 epochs. Let's update the model.fit method to automatically stop training when the validation score doesn't improve.
#We'll use a callback that tests a training condition for every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.
#https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/callbacks/EarlyStopping
model = build_model()

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

#The graph shows the average error is about \$2,500 dollars. Is this good? Well, \$2,500 is not an insignificant amount when some of the labels are only $15,000.
#Let's see how did the model performs on the test set:


[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

#Predict
#Finally, predict some housing prices using data in the testing set:
test_predictions = model.predict(test_data).flatten()

print("Predict completed. test_predictions:  " )
print(test_predictions)

#This notebook introduced a few techniques to handle a regression problem.
#•Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
#•Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
#•When input data features have values with different ranges, each feature should be scaled independently.
#•If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
#•Early stopping is a useful technique to prevent overfitting.




