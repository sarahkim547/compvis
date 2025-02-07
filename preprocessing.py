"""
This file contains all of the functions for pre-processing the data
"""
import argparse
import numpy as np
import os
import random
from skimage import io


PATCH_SIZE = 51
PAD = int((PATCH_SIZE - 1) / 2)


def normalizeImage(image):
    """
    Does color normalization of an image across all 3 channels
    :param image: input image, a numpy array
    :return normalized image:
    """
    return (image.astype(np.float32) - 128) / 128


def extractPatches(pic_file, patch_global_num, patch_sample_num, sample, pic_dir, mask_dir, patch_dir, sample_labels):
    im = io.imread(os.path.join(pic_dir, pic_file))
    im = normalizeImage(im)
    mask = getMask(pic_file, mask_dir)
    h, w = mask.shape
    for i in range(PAD, h - PAD, 2):
        for j in range(PAD, w - PAD, 2):
            if sample is None or patch_global_num in sample:
                assert sample_labels[patch_sample_num] == (mask[i, j] // 127)
                curr_patch = im[i-PAD:i+PAD+1, j-PAD:j+PAD+1, :]
                patch_file = os.path.join(patch_dir, '{:07d}'.format(patch_sample_num))
                np.save(patch_file, curr_patch)
                patch_sample_num += 1
            patch_global_num += 1

    return patch_global_num, patch_sample_num


def getMask(pic_file, mask_dir):
    pic_name, _ = os.path.splitext(pic_file)
    mask_file = 'TM_' + pic_name + '.png'
    return io.imread(os.path.join(mask_dir, mask_file))


def getLabels(mask):
    mask //= 127  # Convert {0, 128, 255} to {0, 1, 2}
    h, w = mask.shape
    labels = mask[PAD:h - PAD:2, PAD:w - PAD:2]
    return labels.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory to find data.')
    parser.add_argument('--num', type=int, default=None, help='Number of examples to process, sampled uniformly.')
    parser.add_argument('--num0', type=int, default=None, help='Number of examples of class 0 to process.')
    parser.add_argument('--num1', type=int, default=None, help='Number of examples of class 1 to process.')
    parser.add_argument('--num2', type=int, default=None, help='Number of examples of class 2 to process.')
    args = parser.parse_args()

    pic_dir = os.path.join(args.data_dir, 'pics')
    mask_dir = os.path.join(args.data_dir, 'masks')
    patch_dir = os.path.join(args.data_dir, 'patches')
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)

    print('Processing labels')
    all_labels = []
    for pic_file in sorted(os.listdir(pic_dir)):
        mask = getMask(pic_file, mask_dir)
        labels = getLabels(mask)
        all_labels.append(labels)
    all_labels = np.concatenate(all_labels)

    if args.num is None and args.num0 is None:
        sampled_labels = all_labels
        sample = None
    elif args.num is not None:
        sample = random.sample(range(len(all_labels)), args.num)
        sample.sort()
        sampled_labels = all_labels[sample]
        np.save(os.path.join(args.data_dir, 'labels'), sampled_labels)
        sample = set(sample)
    else:
        sample = []
        for label, num in [(0, args.num0), (1, args.num1), (2, args.num2)]:
            indices = list(np.argwhere(all_labels == label).flatten())
            sample.append(random.sample(indices, num))
        sample = np.concatenate(sample)
        sample.sort()
        sampled_labels = all_labels[sample]
        np.save(os.path.join(args.data_dir, 'labels'), sampled_labels)
        sample = set(sample)
    np.save(os.path.join(args.data_dir, 'labels'), sampled_labels)

    patch_global_num, patch_sample_num = 0, 0
    for pic_file in sorted(os.listdir(pic_dir)):
        print('Processing image: {}'.format(pic_file))
        patch_global_num, patch_sample_num = extractPatches(pic_file, patch_global_num, patch_sample_num, sample,
                                                            pic_dir, mask_dir, patch_dir, sampled_labels)


if __name__ == '__main__':
    main()
