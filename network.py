import argparse
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical
import numpy as np
import os


NUM_TRAIN_EXAMPLES = 491220
NUM_TEST_EXAMPLES = 517361
BATCH_SIZE = 20
LEARNING_RATE = 0.01
DECAY_EPOCHS = 10


class DataGenerator(Sequence):

    def __init__(self, patch_dir, labels, num_examples, batch_size):
        self.patch_dir = patch_dir
        self.labels = labels
        self.num_examples = num_examples
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.num_examples / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_x = []
        for i in range(start_idx, end_idx):
            if i < self.num_examples:
                patch_file = os.path.join(self.patch_dir, '{:07d}.npy'.format(i))
                patch = np.load(patch_file)
                batch_x.append(patch)

        batch_x = np.stack(batch_x)
        batch_y = self.labels[start_idx:end_idx]

        return batch_x, batch_y


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=25, kernel_size=4, activation='relu', input_shape=(51, 51, 3)))
    # model.add(Dropout(rate=0.1))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=50, kernel_size=5, activation='relu'))
    # model.add(Dropout(rate=0.2))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=80, kernel_size=6, activation='relu'))
    # model.add(Dropout(rate=0.25))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(units=1024, activation='relu'))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    model.reset_states()

    def step_decay(epoch, lr):
        if epoch != 0 and epoch % DECAY_EPOCHS == 0:
            return lr / 2
        return lr

    return model, LearningRateScheduler(step_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, required=True, help='Number of training examples.')
    parser.add_argument('--num_test', type=int, required=True, help='Number of testing examples.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--train_dir', default='train_data')
    parser.add_argument('--test_dir', default='test_data')
    args = parser.parse_args()
    if args.num_train < 1 or args.num_test < 1:
        raise ValueError('num_train and num_test must be positive integers.')
    if args.epochs < 1:
        raise ValueError('Epochs must be a positive integer.')

    model, lr_schedule = create_model()
    train_labels = to_categorical(np.load(os.path.join(args.train_dir, 'labels.npy')), num_classes=2)
    test_labels = to_categorical(np.load(os.path.join(args.test_dir, 'labels.npy')), num_classes=2)
    train_batch_generator = DataGenerator(os.path.join(args.train_dir, 'patches'),
                                          train_labels, args.num_train, BATCH_SIZE)
    test_batch_generator = DataGenerator(os.path.join(args.test_dir, 'patches'),
                                         test_labels, args.num_test, BATCH_SIZE)

    model.fit_generator(generator=train_batch_generator,
                        epochs=args.epochs,
                        shuffle=True,
                        verbose=1,
                        validation_data=test_batch_generator,
                        # callbacks=[lr_schedule],
                        # class_weight={0: 1.0, 1: 1.6, 2: 1.0},
                        # use_multiprocessing=True,
                        # workers=8,
                        )

    model_folder = 'models'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model.save(os.path.join(model_folder, 'model.h5'))

if __name__ == '__main__':
    main()
