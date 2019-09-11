#Model progress can be saved during—and after—training. This means a model can resume where it left off and avoid long training times. Saving also means you can share your model and others can recreate your work.
#When publishing research models and techniques, most machine learning practitioners share:
# •code to create the model, and
# •the trained weights, or parameters, for the model

# Sharing this data helps others understand how the model works and try it themselves with new data.
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

#Get an example dataset
#To speed up these demonstration runs, only use the first 1000 examples:

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


#Define a model
#Let's build a simple model we'll use to demonstrate saving and loading weights.
# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model


# Create a basic model instance
model = create_model()
model.summary()

#Save checkpoints during training
#The primary use case is to automatically save checkpoints during and at the end of training. This way you can use a trained model without having to retrain it,
#or pick-up training where you left of—in case the training process was interrupted.
#tf.keras.callbacks.ModelCheckpoint is a callback that performs this task. The callback takes a couple of arguments to configure checkpointing.
#Checkpoint callback usage
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir =  os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=0)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback], verbose=0)  # pass callback to training
print("Create checkpoint callback completed")

#Create a new, untrained model. When restoring a model from only weights, you must have a model with the same architecture as the original model.
#Since it's the same model architecture, we can share weights despite that it's a different instance of the model.
#Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):

model = create_model()

loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#Then load the weights from the checkpoint, and re-evaluate:
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#Checkpoint callback options

#The callback provides several options to give the resulting checkpoints unique names, and adjust the checkpointing frequency.
#Train a new model, and save uniquely named checkpoints once every 5-epochs:
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=0,
                                                # Save weights, every 5-epochs.
                                                period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

print("Checkpoint save for every 5-epochs completed ")
#Now, have a look at the resulting checkpoints (sorting by modification date):
import pathlib

# Sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
print("sort checkpoints completed " , latest)

#Note: the default tensorflow format only saves the 5 most recent checkpoints.

#To test, reset the model and load the latest checkpoint:
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model latest, accuracy: {:5.2f}%".format(100*acc))
   
#What are these files?
#The above code stores the weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format. Checkpoints contain:
# •One or more shards that contain your model's weights. 
# •An index file that indicates which weights are stored in a which shard. 

#If you are only training a model on a single machine, you'll have one shard with the suffix: .data-00000-of-00001

#Manually save weights
#Above you saw how to load the weights into a model.

#Manually saving the weights is just as simple, use the Model.save_weights method
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')
print("Restored model save completed")

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels, verbose=0)
print("Restored model manual load_weights, accuracy: {:5.2f}%".format(100*acc))

#Save the entire model 

#The entire model can be saved to a file that contains the weight values, the model's configuration, and even the optimizer's configuration.
#This allows you to checkpoint a model and resume training later—from the exact same state—without access to the original code.
#Saving a fully-functional model in Keras is very useful—you can load them in TensorFlow.js and then train and run them in web browsers.
#Keras provides a basic save format using the HDF5 standard. For our purposes, the saved model can be treated as a single binary blob.

model = create_model()

model.fit(train_images, train_labels, epochs=5, verbose=0)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

#Check its accuracy:
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model manual saved, accuracy: {:5.2f}%".format(100*acc))

#This technique saves everything:
#•The weight values
#•The model's configuration(architecture)
#•The optimizer configuration

#Keras saves models by inspecting the architecture. Currently, it is not able to save TensorFlow optimizers (from tf.train).
#When using those you will need to re-compile the model after loading, and you will loose the state of the optimizer.





