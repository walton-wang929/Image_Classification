# How to fine tune a image classifer on our own dataset 

### step1: build image data

*Converts image data to TFRecords file format with Example protos*

The image data set is expected to reside in JPEG files located in the following directory structure.
  * data_dir/label_0/image0.jpeg
  * data_dir/label_0/image1.jpg
  * ...
  * data_dir/label_1/weird-image.jpeg
  * data_dir/label_1/my-image.jpeg

where the sub-directory is the unique label associated with these images.

This TensorFlow provide script [build_image_data.py](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py) converts the training and evaluation data into a sharded data set consisting of TFRecord files
  * train_directory/train-00000-of-01024
  * train_directory/train-00001-of-01024
  * ...
  * train_directory/train-01023-of-01024
  
and

  * validation_directory/validation-00000-of-00128
  * validation_directory/validation-00001-of-00128
  * ...
  * validation_directory/validation-00127-of-00128
  
where we have selected 1024 and 128 shards for each data set. Each record within the TFRecord file is a serialized Example proto. The Example proto contains the following fields:

  * image/encoded: string containing JPEG encoded image in RGB colorspace
  * image/height: integer, image height in pixels
  * image/width: integer, image width in pixels
  * image/colorspace: string, specifying the colorspace, always 'RGB'
  * image/channels: integer, specifying the number of channels, always 3
  * image/format: string, specifying the format, always 'JPEG'
  * image/filename: string containing the basename of the image file e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  * image/class/label: integer specifying the index in a classification layer. The label ranges from [0, num_labels] where 0 is unused and left as the background class.
  * image/class/text: string specifying the human-readable version of the label e.g. 'dog'
  
If your data set involves bounding boxes, please look at [build_imagenet_data.py]() .

###### code:
```
# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=$HOME/my-custom-data/

# build the preprocessing script.
cd tensorflow-models/inception
bazel build //inception:build_image_data

# convert the data.
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8

```
where the $OUTPUT_DIRECTORY is the location of the sharded TFRecords. The $LABELS_FILE will be a text file that is read by the script that provides a list of all of the labels. For instance, in the case flowers data set, the $LABELS_FILE contained the following data:
```
daisy
dandelion
roses
sunflowers
tulips
```
Note that each row of each label corresponds with the entry in the final classifier in the model. That is, the daisy corresponds to the classifier for entry 1; dandelion is entry 2, etc. We skip label 0 as a background class.

Once this data set is built, you are ready to train or fine-tune an Inception model on this data set.

Note, if you are piggy backing on the flowers retraining scripts, be sure to update num_classes() and num_examples_per_epoch() in flowers_data.py to correspond with your data.


### step2: prepare retrained model

download a pre-trained model like so:

```
# location of where to place the Inception v3 model
INCEPTION_MODEL_DIR=$HOME/inception-v3-model
mkdir -p ${INCEPTION_MODEL_DIR}
cd ${INCEPTION_MODEL_DIR}

# download the Inception v3 model
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz

# this will create a directory called inception-v3 which contains the following files.
> ls inception-v3
README.txt
checkpoint
model.ckpt-157585
```

### step3: Retraining 

```
# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
cd tensorflow-models/inception
bazel build //inception:flowers_train

# Path to the downloaded Inception-v3 model.
MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"

# Directory where the flowers data resides.
FLOWERS_DATA_DIR=/tmp/flowers-data/

# Directory where to save the checkpoint and events files.
TRAIN_DIR=/tmp/flowers_train/

# Run the fine-tuning on the flowers data set starting from the pre-trained
# Imagenet-v3 model.
bazel-bin/inception/flowers_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
```

### step4: Evaluation

```
# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
cd tensorflow-models/inception
bazel build //inception:flowers_eval

# Directory where we saved the fine-tuned checkpoint and events files.
TRAIN_DIR=/tmp/flowers_train/

# Directory where the flowers data resides.
FLOWERS_DATA_DIR=/tmp/flowers-data/

# Directory where to save the evaluation events files.
EVAL_DIR=/tmp/flowers_eval/

# Evaluate the fine-tuned model on a hold-out of the flower data set.
bazel-bin/inception/flowers_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once
```
