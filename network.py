from keras.layers import Conv2D, Dense, Dropout, MaxPool2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=25, kernel_size=4, activation='relu'))
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
