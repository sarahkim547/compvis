from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.utils import Sequence, to_categorical
import numpy as np
import os


NUM_TRAIN_EXAMPLES = 491220
NUM_TEST_EXAMPLES = 517361
BATCH_SIZE = 20
NUM_EPOCHS = 1


class DataGenerator(Sequence):

    def __init__(self, patch_dir, labels, batch_size):
        self.patch_dir = patch_dir
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(NUM_TRAIN_EXAMPLES / float(self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_x = []
        for i in range(start_idx, end_idx):
            if i < NUM_TRAIN_EXAMPLES:
                patch_file = os.path.join(self.patch_dir, '{:07d}.npy'.format(i))
                batch_x.append(np.load(patch_file))

        batch_x = np.stack(batch_x)
        batch_y = self.labels[start_idx:end_idx]

        return batch_x, batch_y


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
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def main():
    model = create_model()
    train_labels = to_categorical(np.load('train_data/labels.npy'))
    test_labels = to_categorical(np.load('test_data/labels.npy'))
    train_batch_generator = DataGenerator('train_data/patches', train_labels, BATCH_SIZE)
    test_batch_generator = DataGenerator('test_data/patches', test_labels, BATCH_SIZE)

    model.fit_generator(generator=train_batch_generator,
                        steps_per_epoch=(NUM_TRAIN_EXAMPLES // BATCH_SIZE),
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        validation_data=test_batch_generator,
                        validation_steps=(NUM_TEST_EXAMPLES // BATCH_SIZE),
                        use_multiprocessing=True,
                        workers=16)


if __name__ == '__main__':
    main()
