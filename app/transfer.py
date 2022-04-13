import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
from tkinter import ttk
import pathlib
from pathlib import Path
from tkinter.filedialog import askopenfilename
from app.functions import load_img, imshow
from app.functions import StyleAndContentExtractor, train_step, loss
from IPython.display import Image
Image("/content/result.png")

# vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# vgg.summary()

style_weight = 100.0
content_weight = 5.0
tv_weight = 0.1

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
content_layers = ['block4_conv2']

extractor = StyleAndContentExtractor(style_layers=style_layers, content_layers=content_layers)
sample_image = np.ones((1, 512, 512, 3), dtype=np.float32)
style_and_content_targets = extractor(sample_image)

# ttk().withdraw()
content_path = askopenfilename()
print(content_path)
# content_path = '/moscow-moscow-city-moskva-moskva-siti-gorod-city.jpeg'
style_path = '/Users/user/Desktop/projects/External/style_transfer/style_Image/abstraction_1.jpeg'
print(style_path)

content_image = load_img(content_path, 1024)
style_image = load_img(style_path, 1024)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)
opt = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.99, epsilon=1e-2)

start = time.time()
epochs = 1
steps_per_epoch = 50

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image, loss_func=loss, optimizer=opt, style_targets=style_targets, content_targets=content_targets,
               style_weight=style_weight, content_weight=content_weight, tv_weight=tv_weight, extractor=extractor)
plt.imsave("result.png", image.numpy()[0])
end = time.time()