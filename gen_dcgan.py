import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image

rnd_len = 100
img_count = 4#1170

labels = ["0", "1"]
for label in labels:
    model_path = f"./data/dcgan/generator_{label}.h5"
    image_path = f"./data/dcgan/{label}"
    model = keras.models.load_model(model_path)
    noise = np.random.normal(0, 1, (img_count, rnd_len))
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    for i in range(gen_imgs.shape[0]):
        img_save_loc = f"{image_path}/gen_{str(i+1).zfill(5)}.png"
        plt.imsave(img_save_loc, gen_imgs[i, :, :, 0], cmap="gray")
        img = Image.open(img_save_loc).convert('RGB')
        img.save(img_save_loc)
