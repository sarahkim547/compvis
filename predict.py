import argparse
from keras.models import load_model
import numpy as np
import os
from skimage import io

from preprocessing import normalizeImage, PATCH_SIZE, PAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    pic_dir = os.path.join(args.data_dir, 'pics')
    pred_dir = os.path.join(args.data_dir, 'predictions')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for pic_file in sorted(os.listdir(pic_dir)):
        print('Processing image: {}'.format(pic_file))
        im = io.imread(os.path.join(pic_dir, pic_file))
        im = normalizeImage(im)
        h, w = im.shape[:2]
        pred_ar = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(PAD, h - PAD):
            for j in range(PAD, w - PAD):
                patch = im[i-PAD:i+PAD+1, j-PAD:j+PAD+1, :]
                patch = patch.reshape((1, PATCH_SIZE, PATCH_SIZE, 3))
                pred = model.predict_proba(patch).flatten()
                pred_ar[i, j, :] = pred

        pic_name, _ = os.path.splitext(pic_file)
        np.save(os.path.join(pred_dir, pic_name), pred_ar)


if __name__ == '__main__':
    main()
