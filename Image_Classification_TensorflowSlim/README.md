# Overview

This repo use TFSlim as base to realize training a Image_Classification network.


* In `datasets/`, including some codes to convert image file to TFrecord file.
* In `nets/`, including common net structure(alexnet, inception, vgg, resnet and the like)
* In `preprocessing/`, including some image preprocessing methods.
* In `scripts/`, including all shell scripts for training and evaluating.

# How to use?
* Goto `scripts/`, find training script and then excute it. you can change parameters directly in shell script.  

* parameters can modify:
  * Optimizer
  * Learning Rate
  * Fine-Tune
  * Dataset directory
  * GPU and CPU configuration

* steps:
  1. prepare training and validation, test datasets
  2. convert data format to TFrecord
  3. select fine-tuned model weights / or train from stratch
  4. determine training parameters
  5. run shell script to train 
