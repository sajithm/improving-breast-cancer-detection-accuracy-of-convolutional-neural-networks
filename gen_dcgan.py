import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image

rnd_len = 100
img_count = 1170

model_dir = "./models/dcgan"
image_dir = "./data/dcgan"
assert os.path.exists(model_dir), f"{model_dir} does not exist"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

labels = ["0", "1"]
for label in labels:
    model_path = os.path.join(model_dir, f"generator_{label}.h5")
    assert os.path.exists(model_path), f"{model_path} does not exist"
    image_path = os.path.join(image_dir, f"{label}")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    model = keras.models.load_model(model_path, compile=False)
    noise = np.random.normal(0, 1, (img_count, rnd_len))
    gen_imgs = model.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    for i in range(gen_imgs.shape[0]):
        img_save_loc = os.path.join(image_path, f"dcgan_{str(i+1).zfill(5)}.png")
        plt.imsave(img_save_loc, gen_imgs[i, :, :, 0], cmap="gray")
        img = Image.open(img_save_loc).convert('RGB')
        img.save(img_save_loc)
