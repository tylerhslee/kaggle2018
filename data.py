# -*- coding: utf-8 -*-

import cv2

import numpy as np

from logger import Logger

_ = Logger(__name__).add_stream_handler('info')  \
                    .add_file_handler('data.log', 'debug')  \
                    .create()


class Image(object):

    """
    A container for the original images.
    It has methods that can generate individual cell images
    and preprocess any images for training/testing.
    """

    def __init__(self, id_, cv2_img, masks):
        self.id = id_
        self.img = cv2_img
        self.masks = masks

    def add_padding(self, img, cell_width, cell_height):
        """
        Makes sure that all individual cell images that comes from a single
        image ID has the same dimensions by adding black paddings.
        Note that cell images from different image IDs may not have the same
        dimensions.

        image: cv2 image object
        cell_width: target width of this cell image
        cell_height: target height of this cell image
        """
        image = img.astype(np.uint8)
        image_width = image.shape[0]
        image_height = image.shape[1]
        diff_width = cell_width - image_width
        diff_height = cell_height - image_height
        pad_width = [0] * diff_width
        pad_height = [0] * image_width

        padded_image = [list(r) + pad_width for r in image]
        for i in range(diff_height):
            padded_image.append(pad_height)
        Image.display(np.array(padded_image))
        return np.array(padded_image)
    
    def preprocess(self):
        return self

    @classmethod
    def display(cls, cv2_img, name='Image'):
        # Press any key to close window
        cv2.imshow(name, cv2_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def write(cls, cv2_imgs, template=None):
        """
        Saves a list of images to the disk.
        The name of each image is determined by an enumeration appended to
        the end of the template supplied as a parameter.
        """
        for i, img in enumerate(cv2_imgs):
            file_name = '{}_{}.png'.format(template, i + 1)
            cv2.imwrite(file_name, img)

    @property
    def cells(self):
        """
        Using masks, extract an image of each cell from the original image
        and normalize the dimensions of all resulting images.
        """
        _.info('Extracting cells from Image ID %s' % self.id)
        widths = [m.shape[0] for m in self.masks]
        heights = [m.shape[1] for m in self.masks]
        cell_width = max(widths)
        cell_height = max(heights)
        nmasks = [self.add_padding(m, cell_width, cell_height) for m in self.masks]
        return nmasks
