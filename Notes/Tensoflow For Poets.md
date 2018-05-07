# simple transfer learning tensorflow training classifer (last layer bottleneck)
run TensorFlow on a single machine, and will train a simple classifier to classify images of flowers.

We will be using transfer learning, which means we are starting with a model that has been already trained on another problem. We will then be retraining it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order.

We are going to use a model trained on the ImageNet Large Visual Recognition Challenge dataset. These models can differentiate between 1,000 different classes, like Dalmatian or dishwasher. You will have a choice of model architectures, so you can determine the right tradeoff between speed, size and accuracy for your problem.

### What you'll Learn
- [x] How to use Python and TensorFlow to train an image classifier
- [x] How to classify images with your trained classifier

### step1: Preparations

* Install TensorFlow
 
 Before we can begin the tutorial you need to install tensorflow.

> If you already have TensorFlow installed, be sure it is a recent version. This codelab requires at least version 1.2. You can upgrade to the most recent stable branch with

> pip install --upgrade tensorflow

* Clone the git repository

Clone the repository and cd into it. This is where we will be working.

```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2
```
### step2: dataset(using your own dataset or downloading provided sample dataset)

Before you start any training, you'll need a set of images to teach the model about the new classes you want to recognize. We've created an archive of creative-commons licensed flower photos to use initially. Download the photos (218 MB) by invoking the following two commands:
```
curl http://download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C tf_files
```
You should now have a copy of the flower photos in your working directory. Confirm the contents of your working directory by issuing the following command:
```
ls tf_files/flower_photos
```
The preceding command should display the following objects:
```
daisy/
dandelion/
roses/
sunflowers/
tulip/
LICENSE.txt
```
### step3: Configure your network

The retrain script can retrain either Inception V3 model or a MobileNet. In this exercise, we will use a MobileNet. The principal difference is that Inception V3 is optimized for accuracy, while the MobileNets are optimized to be small and efficient, at the cost of some accuracy.

Inception V3 has a first-choice accuracy of 78% on ImageNet, but is the model is 85MB, and requires many times more processing than even the largest MobileNet configuration, which achieves 70.5% accuracy, with just a 19MB download.

Pick the following configuration options:

> * Input image resolution: 128,160,192, or 224px. Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy. We recommend 224 as an initial setting.
> * The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25. We recommend 0.5 as an initial setting. The smaller models run significantly faster, at a cost of accuracy.

With the recommended settings, it typically takes only a couple of minutes to retrain on a laptop. You will pass the settings inside Linux shell variables. Set those shell variables as follows:
```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
```
### step4: Retarining 
As noted in the introduction, Imagenet models are networks with millions of parameters that can differentiate a large number of classes. We're only training the final layer of that network, so training will end in a reasonable amount of time.

Start your retraining with one big command (note the --summaries_dir option, sending training progress reports to the directory that tensorboard is monitoring) :
```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos
```
This script downloads the pre-trained model, adds a new final layer, and trains that layer on the flower photos you've downloaded. 

The first retraining command iterates only 500 times. You can very likely get improved results (i.e. higher accuracy) by training for longer. To get this improvement, remove the parameter --how_many_training_steps to use the default 4,000 iterations.
```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models/"${ARCHITECTURE}" \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos
```
### step5: Tensorboard to check
Before starting the training, launch tensorboard in the background. TensorBoard is a monitoring and inspection tool included with tensorflow. You will use it to monitor the training progress.
```
tensorboard --logdir tf_files/training_summaries &
```

### More about Bottlenecks

*This section and the next provide background on how this retraining process works.*

The first phase analyzes all the images on disk and calculates the bottleneck values for each of them. What's a bottleneck?

These ImageNet models are made up of many layers stacked on top of each other, a simplified picture of Inception V3 from TensorBoard, is shown above (all the details are available in this [paper](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), with a complete picture on page 6). These layers are pre-trained and are already very valuable at finding and summarizing information that will help classify most images. For this codelab, you are training only the last layer (final_training_ops in the figure below). While all the previous layers retain their already-trained state.
![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/84a6154ed64fd0fb.png)

In the above figure, the node labeled "softmax", on the left side, is the output layer of the original model. While all the nodes to the right of the "softmax" were added by the retraining script.

A bottleneck is an informal term we often use for the layer just before the final output layer that actually does the classification. "Bottelneck" is not used to imply that the layer is slowing down the network. We use the term bottleneck because near the output, the representation is much more compact than in the main body of the network.

Every image is reused multiple times during training. Calculating the layers behind the bottleneck for each image takes a significant amount of time. Since these lower layers of the network are not being modified their outputs can be cached and reused.

So the script is running the constant part of the network, everything below the node labeled Bottlene... above, and caching the results.

The command you ran saves these files to the bottlenecks/ directory. If you rerun the script, they'll be reused, so you don't have to wait for this part again.

### step6: Using the Retrained Model

The retraining script writes data to the following two files:

> tf_files/retrained_graph.pb, which contains a version of the selected network with a final layer retrained on your categories.
> tf_files/retrained_labels.txt, which is a text file containing labels.

#### Classifying an image
The codelab repo also contains a copy of tensorflow's label_image.py example, which you can use to test your network. Take a minute to read the help for this script:
```
python -m  scripts.label_image -h
```
As you can see, this Python program takes quite a few arguments. The defaults are all set for this project, but if you used a MobileNet architecture with a different image size you will need to set the --input_size argument using the variable you created earlier: --input_size=${IMAGE_SIZE}.

runing:
```
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```
You might get results like this for a daisy photo:

```
daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)
```
This indicates a high confidence (~99%) that the image is a daisy, and low confidence for any other label.

You can use label_image.py to classify any image file you choose, either from your downloaded collection, or new ones. You just have to change the --image file name argument to the script.

### Trying Other Hyperparameters(optional)
The retraining script has several other command line options you can use.

You can read about these options in the help for the retrain script:
```
python -m scripts.retrain -h
```
Try adjusting some of these options to see if you can increase the final validation accuracy.

For example, the --learning_rate parameter controls the magnitude of the updates to the final layer during training. So far we have left it out, so the program has used the default learning_rate value of 0.01. If you specify a small learning_rate, like 0.005, the training will take longer, but the overall precision might increase. Higher values of learning_rate, like 1.0, could train faster, but typically reduces precision, or even makes training unstable.

You need to experiment carefully to see what works for your case.
