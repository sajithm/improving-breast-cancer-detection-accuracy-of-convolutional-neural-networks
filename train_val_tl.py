import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, callbacks
import time, json, os, shutil, argparse
import warnings
warnings.simplefilter("ignore")

labels = ["0", "1"]
data_dir_validation = "./data/validation"
img_width, img_height, img_depth = 320, 320, 1
tli_width, tli_height, tli_depth = 224, 224, 3

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", help="Source Folder name(s)", type=str, required=True)
parser.add_argument("-d", "--data_dir", help="Data Folder path", default="./data", type=str, required=False)
parser.add_argument("-m", "--model_dir", help="Path to store models and checkpoints", default="./models_tl", type=str, required=False)
parser.add_argument("-e", "--epochs", help = "Epochs", default=250, type=int, required=False)
parser.add_argument("-b", "--batch_size", help = "Batch Size", default=26, type=int, required=False)
args = parser.parse_args()
params = vars(args)
assert params["source"], "source folder(s) is required"
epochs = params["epochs"]
batch_size = params["batch_size"]
data_dir = params["data_dir"]
model_dir = params["model_dir"]

input_dirs = [s.strip() for s in params["source"].split(',')]
model_name = '+'.join(input_dirs)
model_path = os.path.join(model_dir, model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

data_dir_training = os.path.join(data_dir, model_name)
path_history = os.path.join(model_path, f'history_{batch_size}.csv')
path_model = os.path.join(model_path, f'model_{batch_size}.h5')
path_ckpt = os.path.join(model_path, f'checkpoints_{batch_size}')
if not os.path.exists(path_ckpt):
    os.makedirs(path_ckpt)

datasources = [os.path.join(data_dir, d) for d in input_dirs]
if len(datasources) > 1:
    if not os.path.exists(data_dir_training):
        os.makedirs(data_dir_training)

    for label in labels:
        path_dest = os.path.join(data_dir_training, label)
        if not os.path.exists(path_dest):
            os.makedirs(path_dest)
        if len(os.listdir(path_dest)) == 0:
            for datasource in datasources:
                assert os.path.exists(datasource), f"{datasource} doesn't exist"
                path_src = os.path.join(datasource, label)
                assert os.path.exists(path_src), f"{path_src} doesn't exist"
                for f in os.listdir(path_src):
                    shutil.copy(os.path.join(path_src, f), path_dest)

training_samples = 2 * 1170 * len(input_dirs)
validation_samples = 2 * 130

resnet_50 = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
resnet_50.trainable=False

#inputs = tf.keras.Input(shape=(img_width, img_height, img_depth))
#inputs = tf.keras.layers.Conv2D(3,(3,3),padding='same')(inputs)
inputs = tf.keras.Input(shape=(tli_width, tli_height, tli_depth))
x = resnet_50(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(1)(x)
outputs = tf.keras.layers.Activation('sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_TL")
#model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    data_dir_training,
    target_size=(tli_width, tli_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=42)

validation_generator = test_datagen.flow_from_directory(
    data_dir_validation,
    target_size=(tli_width, tli_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=42)

callback_log = callbacks.CSVLogger(path_history, append=True)
callback_model_ckpt = callbacks.ModelCheckpoint(filepath=path_ckpt + "/model_{epoch:03d}.ckpt", save_weights_only=False, save_freq=50 * (training_samples // batch_size)) 
callback_model_save = callbacks.ModelCheckpoint(filepath=path_model, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)

last_epoch = 0
checkpoints = sorted([f.path for f in os.scandir(path_ckpt) if f.is_dir()], reverse=True)
if len(checkpoints) > 0:
    checkpoint = checkpoints[0]
    last_epoch = int(list(reversed(checkpoint.split('_')))[0].replace(".ckpt", ""))
    model = tf.keras.models.load_model(checkpoint)

time_in = time.time()  # record using time start

hist = model.fit(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs,
    initial_epoch = last_epoch,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    callbacks=[callback_log, callback_model_ckpt, callback_model_save])

time_out = time.time()
print('\n', 'Time cost:', '\n', time_out-time_in)