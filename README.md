# Kaggle Challenge - 2018 Data Science Bowl

Details [here](https://www.kaggle.com/c/data-science-bowl-2018#description).

This challenge requires us to identify the pixels that belong to a "cell" in rectangular masks that vary in dimensions. Each mask contains dozens of cells, and our script must be able to identify all instances of cells and their corresponding pixels.

## Installation
```bash
$ git clone https://github.com/tylerhslee/kaggle2018
$ cd kaggle2018
$ pip3 install -r requirements.txt
$ chmod +x main.py
$ ./main.py
```

## Method
I plan to take the following steps to approach this problem.

  1. Isolate each instance of cell image from every mask and save them as separate .png files.
  
  2. Train [Mask R-CNN model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md) to implement Tensorflow's object segmentation algorithm. (Note to self: [Tutorial](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) to follow)
  
  3. Run the instance segmentation model on the masks originally given to us.

Currently, the project has completed step #1, and is now preparing to work on #2. I'll shortly update the script to generate correct labels for each mask that can be used for training the model.
