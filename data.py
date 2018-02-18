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

    @classmethod
    def normalize(cls, image, size):
        """
        Makes sure that all individual cell images that comes from a single
        image ID has the same dimensions by cutting out a rectangular area
        from the mask that corresponds to the expected size of the cell image.
        Note that cell images from different image IDs may not have the same
        dimensions.

        image: cv2 image object
        size: a tuple of width by height
        """
        image = image.astype(np.uint8)
        area = size[0] * size[1]
        image_width = image.shape[0]
        image_height = image.shape[1]

        cover_row = [[1] * size[0]]
        cover = np.repeat(cover_row, size[1], axis=0).astype(np.uint8)

        # Find the optimal starting point for the rectangle.
        # If too far, the search process will take unnecessarily long.
        # But if too close, it won't cover the entire cell.
        init = np.where(image > 0)
        init_x = max(0, init[1][0] - size[0])
        init_y = max(0, init[0][0] - size[1])
        fin_x = min(init_x + size[0] * 2, image_width - size[0] + 1)
        fin_y = min(init_y + size[1] * 2, image_height - size[1] + 1)

        for i in range(init_x, fin_x):
            for j in range(init_y, fin_y):
                if i % 20 == 0 and j % 20 == 0:
                    _.info('Iteration: (%d, %d)' % (i, j))
                image[j: j + size[1], i: i + size[0]] += cover

                # Test if the rectangle covers the entire cell.
                # If it doesn't, then the number of pixels that are not 0
                # (or black) will be greater than the area of the rectangle.
                if np.sum(image > 0) == area:
                    return image > 0
                image[j: j + size[1], i: i + size[0]] -= cover

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

    def preprocess(self):
        return self

    @property
    def cells(self):
        """
        Using masks, extract an image of each cell from the original image
        and normalize the dimensions of all resulting images.
        """
        _.info('Extracting cells from Image ID %s' % self.id)
        mask_max = 255
        masks = [m == mask_max for m in self.masks]
        full_images = [self.img * m for m in masks]
        widths = [sum(row) for mask in masks for row in mask]
        heights = [len([m[m] for m in mask if len(m[m]) > 0])
                   for mask in masks]

        cell_width = max(widths)
        cell_height = max(heights)
        nmasks = [Image.normalize(m, (cell_width, cell_height)) for m in masks]
        cells = []
        for i, im in enumerate(full_images):
            # 256 is silently converted to 0
            im -= im == mask_max
            cell = im + nmasks[i]
            cells.append(cell)

        fin_cells = np.array([[i[i > 0] for i in cell if len(i[i > 0]) > 0]
                              for cell in cells])

        # Return a list of cell images
        return fin_cells
