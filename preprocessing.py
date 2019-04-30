"""
This file contains all of the functions for pre-processing the data
"""
import numpy as np
import cv2
import PIL
import os
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import io, img_as_ubyte, img_as_float32



def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))

def normalizeImage(image):
    """
    Does color normalization of an image across all 3 channels
    :param image: input image, a numpy array
    :return normalized image:
    """
    return (image.astype(float) - 128) / 128

def extractPatches(im, mask):
    patches = extract_patches_2d(im, (51,51)) 
    pad = int((51 - 1)/2)
    h, w = mask.shape
    labels = mask[pad:h-pad, pad:w-pad]
    labels = labels.flatten()
    return patches, labels


def main():
    # First we import a test image
    # im = cv2.imread('Tissue images/TCGA-18-5592-01Z-00-DX1.tif', -1)
    # im = normalizeImage(im)
    # print(np.mean(im, axis=(0, 1)))

    pic_dir = 'train_data/pics/'
    mask_dir = 'train_data/masks/'

    all_patches = []
    all_labels = []

    for file_name in os.listdir(pic_dir):
        im = load_image(pic_dir + file_name)
        im = normalizeImage(im)

        mask_name = mask_dir + 'TM_' + file_name
        mask = (load_image(mask_name)*2).astype(int)

        patches, labels = extractPatches(im, mask)
        if len(all_patches) == 0:
            all_patches = patches
            all_labels = labels
        else:
            all_patches = np.concatenate((all_patches, patches), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis = 0)
    
     





if __name__ == '__main__':
    main()
