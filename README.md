# Overview

This repo contains three methods for training and deploying a image classification task.

* In `Image_Classfication_Keras/`, use keras to train a network, including training, testing codes.
* In `Image_Classfication_Mobile`, use TFmobile/TFLite to train, trained network can deploy on mobile devices.
* In `Image_Classification_TensorflowSlim`, use TFSlim as a base to train, focus on realize high accurate precision, run by Personal PC / server.

Usually, there are several steps for a Image_Classfication_work.

## step1: data collection and argumentation
  * open source data: There are many open source dataset from the world. like ImageNet, COCO, PASVAL, google Open Images, and some kaggle public competition dataset.
  * collected by yourself data
  * [data argumentation](https://github.com/walton-wang929/Image_Classification/blob/master/Notes/data%20argumentation.md)

## step2: pretrained model comparison and selection
  * [tensoflow provided classication Pre-trained Models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
  * according to your application, make tradeoff between accuracy and speed
  * consider intialization and generalization :thumbsup::thumbsup::thumbsup:

## step3: training
  * [simple fine-tune actively training on Keras]()
  * [TensorFlow for Poets: How to train a custom image recognition model](https://github.com/walton-wang929/Image_Classification/blob/master/Notes/Tensoflow%20For%20Poets.md)
  * [TensorFlow for Poets 2: Optimize for Mobile]()
  * [TensorFlow for Poets 2: TFLite]()
  * [How to fine tune a image classifier on Tensorflow Flowers Data](https://github.com/walton-wang929/Image_Classification/blob/master/Notes/fine%20tune%20Flowers%20Dataset.md)
  * [How to fine tune a image classifer on our own dataset](https://github.com/walton-wang929/Image_Classification/blob/master/Notes/fine%20tune%20own%20dataset.md)
  * [TensorFlow-Slim image classification](https://github.com/walton-wang929/Image_Classification/blob/master/Notes/TensorFlow-Slim%20image%20classification.md) :thumbsup::thumbsup::thumbsup:
  * [optimizer selection](https://github.com/walton-wang929/Image_Classification/blob/master/Notes/optimizer.md)

## step4: test trained model
  * calculate numeric metrics(Accuracy, Precision, Recall, F1)
  * determine classification threshold (ROC Curve, PR curve)
  * No Prediction bias


## step5: deployment to mobile or cloud server

* [TF Mobile and TF lite]()
* PC
* server

# Demo
I fine trained a VGG16 gender recognition model based on my own dataset[22000].

![KLIA Airport](https://github.com/walton-wang929/Image_Classification/raw/master/demo/KLIA%20People%20Detection%20and%20Characteristics%20With%20Skeleton.gif)


# reference:
1. [image-classify-server](https://github.com/ccd97/image-classify-server)
2. [GenderClassifierCNN](https://github.com/scoliann/GenderClassifierCNN/blob/master/genderClassification.py)
3. [deep-machine-learning/Retrained-InceptionV3](https://github.com/deep-machine-learning/Retrained-InceptionV3)
4. [tensorflow-for-poets-2: TFlite](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0)
5. [tensorflow-for-poets-2: Optimize for Mobile](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#0)
6. [googlecodelabs/tensorflow-for-poets-2](https://github.com/googlecodelabs/tensorflow-for-poets-2)
7. [tensorflow-for-poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)
8. [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/blob/master/research/slim/README.md)
9. [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/README.md)
