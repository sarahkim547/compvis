import argparse
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


NUM_TRAIN_EXAMPLES = 491220
NUM_TEST_EXAMPLES = 517361
BATCH_SIZE = 20
DECAY_EPOCHS = 5


class DataGenerator(Sequence):

    def __init__(self, patch_dir, labels, batch_size, max_num_examples):
        self.patch_dir = patch_dir
        self.labels = labels
        self.batch_size = batch_size
        self.max_num_examples = max_num_examples

    def __len__(self):
        return int(np.ceil(NUM_TRAIN_EXAMPLES / float(self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_x = []
        for i in range(start_idx, end_idx):
            if i < self.max_num_examples:
                patch_file = os.path.join(self.patch_dir, '{:07d}.npy'.format(i))
                batch_x.append(np.load(patch_file))

        batch_x = np.stack(batch_x)
        batch_y = self.labels[start_idx:end_idx]

        return batch_x, batch_y


class Eval(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: % f — val_precision: % f — val_recall % f' % (_val_f1, _val_precision, _val_recall))
        return


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=25, kernel_size=4, activation='relu', input_shape=(51, 51, 3)))
    model.add(Dropout(rate=0.1))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=50, kernel_size=5, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=80, kernel_size=6, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    def step_decay(epoch, lr):
        if epoch != 0 and epoch % DECAY_EPOCHS == 0:
            return lr / 2
        return lr

    return model, LearningRateScheduler(step_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, default=0, help='Number of examples to train on (0 = all examples).')
    parser.add_argument('--num_test', type=int, default=0, help='Number of examples to test on (0 = all examples).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    args = parser.parse_args()
    if args.num_train < 0 or args.num_test < 0:
        raise ValueError('num_train and num_test must be nonnegative integers.')
    if args.epochs < 1:
        raise ValueError('Epochs must be a positive integer.')
    num_train = NUM_TRAIN_EXAMPLES if args.num_train == 0 else min(args.num_train, NUM_TRAIN_EXAMPLES)
    num_test = NUM_TEST_EXAMPLES if args.num_test == 0 else min(args.num_test, NUM_TEST_EXAMPLES)

    model, lr_schedule = create_model()
    train_labels = to_categorical(np.load(os.path.join('train_data', 'labels.npy')))
    test_labels = to_categorical(np.load(os.path.join('test_data', 'labels.npy')))
    train_batch_generator = DataGenerator(os.path.join('train_data', 'patches'),
                                          train_labels, BATCH_SIZE, NUM_TRAIN_EXAMPLES)
    test_batch_generator = DataGenerator(os.path.join('test_data', 'patches'),
                                         test_labels, BATCH_SIZE, NUM_TEST_EXAMPLES)

    model.fit_generator(generator=train_batch_generator,
                        steps_per_epoch=(num_train // BATCH_SIZE),
                        epochs=NUM_EPOCHS,
                        shuffle=True,
                        verbose=1,
                        validation_data=test_batch_generator,
                        validation_steps=(num_test // BATCH_SIZE),
                        callbacks=[lr_schedule, Eval()],
                        class_weight={0: 1.0, 1: 18.9, 2: 11.5},
                        use_multiprocessing=True,
                        workers=16)


if __name__ == '__main__':
    main()
