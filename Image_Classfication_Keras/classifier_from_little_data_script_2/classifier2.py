# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:02:51 2017

@author: TWang

Using the bottleneck features of a pre-trained network: 90% accuracy in a minute

A more refined approach would be to leverage a network pre-trained on a large dataset. 
Such a network would have already learned features that are useful for most computer vision problems, 
and leveraging such features would allow us to reach a better accuracy than any method that would only rely on the available data.

We will use the VGG16 architecture, pre-trained on the ImageNet dataset --a model previously featured on this blog. 
Because the ImageNet dataset contains several "cat" classes (persian cat, siamese cat...) and many "dog" classes among its total of 1000 classes, 
this model will already have learned features that are relevant to our classification problem. 
In fact, it is possible that merely recording the softmax predictions of the model over our data rather than the bottleneck features would be 
enough to solve our dogs vs. cats classification problem extremely well. 
However, the method we present here is more likely to generalize well to a broader range of problems, including problems featuring classes absent from ImageNet.

Here's what the VGG16 architecture looks like:
    
    Conv block 1 ==> conv block 2 ==> conv block 3 ==>conv block 4 ==> conv block 5 ==> Fully connected classifer

Our strategy will be as follow: 
    
    we will only instantiate the convolutional part of the model, everything up to the fully-connected layers. 
    We will then run this model on our training and validation data once, recording the output (the "bottleneck features" from th VGG16 model: 
    the last activation maps before the fully-connected layers) in two numpy arrays. 
    Then we will train a small fully-connected model on top of the stored features.

The reason why we are storing the features offline rather than adding our fully-connected model directly on top of a frozen convolutional base 
and running the whole thing, is computational effiency. 
Running VGG16 is expensive, especially if you're working on CPU, and we want to only do it once. 
Note that this prevents us from using data augmentation.

"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
import math
from keras import backend as K

# dimensions of our images.
img_width, img_height = 100, 250

top_model_weights_path = 'bottleneck_fc_model_weights.h5'
top_model_path = 'bottleneck_fc_model.h5'

train_data_dir = r'D:\WANG Tao\gender\data\train'
validation_data_dir = r'D:\WANG Tao\gender\data\validation'

nb_train_samples = 90000
nb_validation_samples = 10000
epochs = 30
batch_size = 200

'''Tensforflow and Theno has different img channel order'''
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    
'''
But let's take a look at how we record the bottleneck features using image data generators:
'''
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # load the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    '''train set'''
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, # this means our generator will only yield batches of data, no labels
        shuffle=False) # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
    
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    #return value is (32000, 4, 4, 512)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples//batch_size)
    
    # save the output as a Numpy array
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    '''validation set'''
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples//batch_size)
    
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

'''
We can then load our saved data and train a small fully-connected model:
'''
def train_top_model():
    
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array([0] * 45000 + [1] * 45000)

    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    validation_labels = np.array([0] * 5000 + [1] * 5000)
    
    
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    
    model.save_weights(top_model_weights_path)
    model.save(top_model_path)
    
    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    
    '''figure'''
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


save_bottlebeck_features()
train_top_model()


'''
result:
    We reach a validation accuracy of 0.90-0.91: not bad at all. 
    This is definitely partly due to the fact that the base model was trained on a dataset that already featured dogs and cats (among hundreds of other classes).
'''