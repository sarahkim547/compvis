"""
This file contains all of the functions for pre-processing the data
"""
import numpy as np
import os
from skimage import io


def normalizeImage(image):
    """
    Does color normalization of an image across all 3 channels
    :param image: input image, a numpy array
    :return normalized image:
    """
    return (image.astype(np.float32) - 128) / 128


def extractPatches(im, mask, patch_num):
    patch_size = 51
    pad = int((patch_size - 1) / 2)
    h, w = mask.shape

    labels = mask[pad:h-pad, pad:w-pad]
    labels = labels.flatten()

    for i in range(pad, h - pad, 2):
        for j in range(pad, w - pad, 2):
            curr_patch = im[i-pad:i+pad+1, j-pad:j+pad+1, :]
            np.save('patches/{:07d}'.format(patch_num), curr_patch)
            patch_num += 1

    return labels


def main():
    # First we import a test image
    # im = cv2.imread('Tissue images/TCGA-18-5592-01Z-00-DX1.tif', -1)
    # im = normalizeImage(im)
    # print(np.mean(im, axis=(0, 1)))

    pic_dir = 'train_data/pics/'
    mask_dir = 'train_data/masks/'
    
    patchesDir = 'patches'
    if not os.path.exists(patchesDir):
        os.mkdir(patchesDir)

    all_labels = []
    patch_num = 0
    for file_name in os.listdir(pic_dir):
        print('Processing image: {}'.format(file_name))
        im = io.imread(pic_dir + file_name)
        im = normalizeImage(im)

        mask_name = mask_dir + 'TM_' + file_name
        mask = io.imread(mask_name) // 127

        labels = extractPatches(im, mask, patch_num)
        all_labels.append(labels)
        patch_num += len(labels)

    all_labels = np.concatenate(all_labels)
    np.save('labels', all_labels)


if __name__ == '__main__':
    main()
