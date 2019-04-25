"""
This file contains all of the functions for pre-processing the data
"""
import numpy as np
import cv2
import PIL
import os


def normalizeImage(image):
    """
    Does color normalization of an image across all 3 channels
    :param image: input image, a numpy array
    :return normalized image:
    """
    return (image.astype(float) - 128) / 128


def main():
    # First we import a test image
    im = cv2.imread('Tissue images/TCGA-18-5592-01Z-00-DX1.tif', -1)
    im = normalizeImage(im)
    print(np.mean(im, axis=(0, 1)))


if __name__ == '__main__':
    main()
