# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:46:20 2017

@author: TWang

Fine-tuning the top layers of a a pre-trained network

To further improve our previous result, we can try to "fine-tune" the last convolutional block of the VGG16 model alongside the top-level classifier.
Fine-tuning consist in starting from a trained network, then re-training it on a new dataset using very small weight updates. 
In our case, this can be done in 3 steps:

    step 1 : instantiate the convolutional base of VGG16 and load its weights
    step 2 : add our previously defined fully-connected model on top, and load its weights
    step 3 : freeze the layers of the VGG16 model up to the last convolutional block
    
    order is like this:
        
        Conv block 1 (frozen)==> conv block 2 (frozen)==> conv block 3 (frozen)==>conv block 4 (frozen)==> conv block 5 (fine tune)==> our own Fully connected classifer(fine tune)

Note that:

    1. in order to perform fine-tuning, all layers should start with properly trained weights: 
    for instance you should not slap a randomly initialized fully-connected network on top of a pre-trained convolutional base. 
    This is because the large gradient updates triggered by the randomly initialized weights would wreck the learned weights in the convolutional base. 
    In our case this is why we first train the top-level classifier, and only then start fine-tuning convolutional weights alongside it.
    
    2. we choose to only fine-tune the last convolutional block rather than the entire network in order to prevent overfitting, 
    since the entire network would have a very large entropic capacity and thus a strong tendency to overfit. 
    The features learned by low-level convolutional blocks are more general, less abstract than those found higher-up, 
    so it is sensible to keep the first few blocks fixed (more general features) and only fine-tune the last one (more specialized features).
    
    3. fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer rather than an adaptative learning rate optimizer such as RMSProp. 
    This is to make sure that the magnitude of the updates stays very small, so as not to wreck the previously learned features.
    
"""
import os
import h5py
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import matplotlib.pyplot as plt
import math

#print(keras.__version__)
# OUTPUT: '2.0.3'

# path to the model weights files.
weights_path = 'vgg16_weights.h5'

top_model_weights_path = 'bottleneck_fc_model_weights.h5'

# dimensions of our images.
img_width = 100
img_height = 250

train_data_dir = r'D:\WANG Tao\gender\data\train'
validation_data_dir = r'D:\WANG Tao\gender\data\validation'

nb_train_samples = 90000
nb_validation_samples = 10000

epochs = 30
batch_size = 200

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

'''build the VGG16 network'''
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape = input_shape))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.summary()
print('Model Bulit.')

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    layer = model.layers[k]
    if isinstance(layer, Convolution2D):
        weights[0] = np.array(weights[0])[:, :, ::-1, ::-1]
f.close()
print('Model loaded.')


'''
After instantiating the VGG base and loading its weights, we add our previously trained fully-connected classifier on top:
'''
# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)
model.summary()
#model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

#==============================================================================
# # CREATE AN "REAL" MODEL FROM VGG16
# # BY COPYING ALL THE LAYERS OF VGG16
# new_model = Sequential()
# 
# for l in model.layers:
#     
#     new_model.add(l)
#     
# # CONCATENATE THE TWO MODELS
# new_model.add(top_model)  
#==============================================================================
#model = Model(input= model.input, output= top_model(model.output))

 
'''
We then proceed to freeze all convolutional layers up to the last convolutional block:
'''
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

'''
Finally, we start training the whole thing, with a very slow learning rate:
'''
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('final_model_weights.h5') # always save your weights after training or during training
model.save('final_model.h5')
    
'''figure'''
plt.figure(1)

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


'''
results:
    
This approach gets us to a validation accuracy of 0.94 after 50 epochs. Great success!

Here are a few more approaches you can try to get to above 0.95:

    more aggresive data augmentation
    more aggressive dropout
    use of L1 and L2 regularization (also known as "weight decay")
    fine-tuning one more convolutional block (alongside greater regularization)

'''
