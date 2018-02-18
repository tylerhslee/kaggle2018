#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use pretrained networks to detect cells in the images.

https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

First create a new train data, where each data point represent only ONE cell.
This data is programmatically generated using the image and masks provided.

Then, train the model with this new train data so that all the cells can be
detected in the test data set.
"""
import os
import cv2
import yaml

from data import Image
from logger import Logger

# Make sure to edit the config file to the proper root directory.
# It is the full path to the stage1_train folder.
with open('config.yml', 'r') as rf:
    config = yaml.safe_load(rf)

ROOT_DIR = config['root-dir']

_ = Logger(__name__).create()


def file_path(id_, mask=None):
    if mask is not None:
        return os.path.join(os.getcwd(), id_, 'masks', mask)
    return os.path.join(os.getcwd(), id_, 'images', id_ + '.png')


def read_img(id_, mask=None):
    if mask is not None:
        p = file_path(id_, mask)
    else:
        p = file_path(id_)
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def load_data(id_):
    mask_files = os.listdir(os.path.join(os.getcwd(), id_, 'masks'))
    image = read_img(id_)
    masks = [read_img(id_, mask=i) for i in mask_files]
    return Image(id_, image, masks)


def save_data(image):
    """
    Save individual cell images to the directory of the respective image IDs
    """
    cells = image.cells
    target_dir = os.path.join(ROOT_DIR, image.id, 'cells')
    current_dir = os.getcwd()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    os.chdir(target_dir)
    Image.write(cells, template='cell')
    os.chdir(current_dir)
    return cells


def sample_background(image_entry, size=(25, 25)):
    """
    image_entry is an entry of the list returned by load_data().
    Use the masks to extract the parts of images that do not contain
    pixels that belong to a cell. Then, use padding to normalize the
    image sizes. They must be the same size as cell images.
    """
    return


def preprocess(*data):
    """
    Each datum is an array of images.
    """
    return data


def train_cell_image(train_data):
    """
    original_data is an array of dictionaries returned by load_data().
    """
    # Array of images of individual cells
    cell_images = [save_data(img) for img in train_data]

    # Array of images of background that doesn't contain cells
    background_images = [sample_background(k) for k in train_data]

    cell_images, background_images = preprocess(cell_images, background_images)


if __name__ == '__main__':
    os.chdir(ROOT_DIR)
    img_ids = os.listdir('.')[:2]
    print(img_ids[0])
    train_data = [load_data(i) for i in img_ids]

    train_cell_image(train_data)
