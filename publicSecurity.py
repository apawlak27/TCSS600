# Image Classification for Public Security
# Alex Pawlak
# TCSS 600

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import os

# Set random seeds and environment
os.environ['PHYTHONHASHSEED'] = '0'
seed = 7
tf.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Create Keras session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(session)

# Set Keras backend: tf=tensorflow th=theano
K.set_image_dim_ordering('tf')

# Set image size
img_width, img_height = 200, 200

# Input shape for theano backend
#input_shape = (3, img_width, img_height)

# Input shape for tensorflow backend
input_shape = (img_width, img_height, 3)

# Image file directories
train_data_dir = 'train/'
validation_data_dir = 'validation/'
test_data_dir = 'test/'

# Training data image data generator
train_generator = ImageDataGenerator(rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True).flow_from_directory(train_data_dir,
                                                         target_size=(img_width, img_height),
                                                         classes=['threat', 'safe'], batch_size=5)

# Validation data image data generator
validation_generator = ImageDataGenerator(rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True).flow_from_directory(validation_data_dir,
                                                         target_size=(img_width, img_height),
                                                         classes=['threat', 'safe'], batch_size=10)

# Testing data image data generator
test_generator = ImageDataGenerator(rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True).flow_from_directory(test_data_dir,
                                                        target_size=(img_width, img_height),
                                                        classes=['threat', 'safe'], batch_size=30)

# Build model

model = Sequential()
model.add(Conv2D(25, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(50, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(75, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Adam(lr=.0001)
# optimizer='rmsprop'

#model.summary()

history = model.fit_generator(train_generator, 
                    steps_per_epoch=40, 
                    validation_data=validation_generator, 
                    validation_steps=5, 
                    epochs=20)

# list all data in history (training and validation data results)
#print(history.history.keys())

# Plot training and validation results for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot training and validation results for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Classify test images

#print(test_generator.class_indices)
test_imgs, test_labels = next(test_generator)
test_labels = test_labels[:,0]
#print(test_labels)

predictions = model.predict_generator(test_generator, steps=1)
#print(predictions)

# Create and plot confusion matrix for test classifications
cm = confusion_matrix(test_labels, np.round(predictions[:, 0]))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm_plot_labels = ['threat', 'safe']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
