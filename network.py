from keras.layers import Conv2D, Dense, Dropout, MaxPool2D
from keras.models import Sequential
import numpy as np


NUM_TRAINING_EXAMPLES = 491220
BATCH_SIZE = 20
NUM_EPOCHS = 1


class DataGenerator(list):

    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(NUM_TRAINING_EXAMPLES / float(self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        batch_x = []
        for i in range(start_idx, end_idx):
            if i < NUM_TRAINING_EXAMPLES:
                batch_x.append(np.load('patches/{:07d}.npy'.format(i)))

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
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def main():
    model = create_model()
    labels = np.load('labels.npy')
    training_batch_generator = DataGenerator(labels, BATCH_SIZE)

    model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch=(NUM_TRAINING_EXAMPLES // BATCH_SIZE),
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        # validation_data=my_validation_batch_generator,
                        # validation_steps=(num_validation_samples // batch_size),
                        use_multiprocessing=True,
                        workers=16)


if __name__ == '__main__':
    main()
