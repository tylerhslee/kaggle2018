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

Currently, the project is completed step #1, and is now preparing to work on #2. In order to properly train the model, however, a few requirements must be met:

  * All images contain pixels that has gray-scale intensity of either 0 (black) or 255 (white).
  * All pixels that belong to a cell instance has gray-scale intensity of 255 (white).
  * All images are of the same dimensions.  

# Implementation
## Step 1: Isolate each instance
The entry script, `main.py`, uses the `Image` class defined in `data.py` to extract individual cell images from each mask.

```python
def load_data(id_):
    mask_files = os.listdir(os.path.join(os.getcwd(), id_, 'masks'))
    # Load the image using the cv2 module
    image = read_img(id_)
    masks = [read_img(id_, mask=i) for i in mask_files]
    return Image(id_, image, masks)

img_ids = os.listdir('.')
train_data = [load_data(i) for i in img_ids]
```

The `Image` class encapsulates the preprocessing of the images. For example, the class has a property called `cells` that returns a list of individual cell images that pertains to the mask ID of the instance.

## Step 2: Train Mask R-CNN model
## Step 3: Run the model
