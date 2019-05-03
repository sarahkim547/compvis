"""
This file contains all of the functions for pre-processing the data
"""
import argparse
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
    assert im.shape[:2] == mask.shape
    h, w = mask.shape

    patch_size = 51
    pad = int((patch_size - 1) / 2)
    num_patches = 0
    for i in range(pad, h - pad, 2):
        for j in range(pad, w - pad, 2):
            curr_patch = im[i-pad:i+pad+1, j-pad:j+pad+1, :]
            np.save('patches/{:07d}'.format(patch_num + num_patches), curr_patch)
            num_patches += 1

    labels = mask[pad:h - pad:2, pad:w - pad:2]
    labels = labels.flatten()
    assert num_patches == len(labels)

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['train', 'test'], required=True, help='Process train or test dataset.')
    args = parser.parse_args()

    data_dir = 'train_data' if args.data == 'train' else 'test_data'
    pic_dir = os.path.join(data_dir, 'pics')
    mask_dir = os.path.join(data_dir, 'masks')
    patch_dir = os.path.join(data_dir, 'patches')
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)

    all_labels = []
    patch_num = 0
    for file_name in os.listdir(pic_dir):
        print('Processing image: {}'.format(file_name))
        im = io.imread(os.path.join(pic_dir, file_name))
        im = normalizeImage(im)

        mask_name = os.path.join(mask_dir, 'TM_' + file_name)
        mask = io.imread(mask_name) // 127

        labels = extractPatches(im, mask, patch_num)
        all_labels.append(labels)
        patch_num += len(labels)

    all_labels = np.concatenate(all_labels)
    np.save(os.path.join(data_dir, 'labels'), all_labels)


if __name__ == '__main__':
    main()
