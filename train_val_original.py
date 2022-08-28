from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers, callbacks
import time, json, os
import matplotlib.pyplot as plt

data_dir_training = "./data/original"
data_dir_validation = "./data/validation"
path_history = "./data/original/history.csv"
path_model = "./data/original/model.h5"

img_width, img_height, img_depth = 320, 320, 1

nb_train_samples = 2340
nb_validation_samples = 260
epochs = 750
batch_size = 26

time_in = time.time()  # record using time start

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, img_depth)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    data_dir_training,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    data_dir_validation,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='binary')

callback_log = callbacks.CSVLogger(path_history, append=True)
callback_earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
callback_model_save = callbacks.ModelCheckpoint(filepath=path_model, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)

hist = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[callback_log, callback_earlystop, callback_model_save])

time_out = time.time()
print('\n', 'Time cost:', '\n', time_out-time_in)