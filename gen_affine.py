### Image Augmentation by Affine Transformation

from keras.preprocessing.image import ImageDataGenerator
import time, json, os


# dimensions of our images.
img_width, img_height = 320, 320
batch_size = 26

data_dir_in = "./data/original"
data_dir_out = "./data/affine"

datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='reflect',
                             horizontal_flip=True,
                             vertical_flip=True)

labels = ["0", "1"]
for label in labels:
    generator = datagen.flow_from_directory(
        data_dir_in,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
        classes = [label],
        shuffle=False,
        save_to_dir=os.path.join(data_dir_out, label), save_format='png')
        
    i = 1
    for batch in generator:
        i += 1
        if i > 45:
            break
