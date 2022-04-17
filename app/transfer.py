import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter.filedialog import askopenfilename
from app.functions import load_img
from app.functions import StyleAndContentExtractor, train_step, loss

# window = tk.Tk()
# window.title("Style Transfer")
# window.rowconfigure(0, minsize=800, weight=1)
# window.columnconfigure(1, minsize=800, weight=1)
# ###
# # label = tk.Label(text="Hello, Tkinter", fg="black", width=30, height=30)
# # label.pack()
# window.mainloop()

# STYLE_WEIGHT = 100.0
# CONTENT_WEIGHT = 5.0
# TV_WEIGHT = 0.1
#
# style_layers = ['block1_conv1',
#                 'block2_conv1',
#                 'block3_conv1',
#                 'block4_conv1',
#                 'block5_conv1']
# content_layers = ['block4_conv2']
#
# extractor = StyleAndContentExtractor(style_layers=style_layers, content_layers=content_layers)
# sample_image = np.ones((1, 512, 512, 3), dtype=np.float32)
# style_and_content_targets = extractor(sample_image)
#
# content_path = askopenfilename()
# print(content_path)
# # content_path = '/moscow-moscow-city-moskva-moskva-siti-gorod-city.jpeg'
# style_path = '/Users/user/Desktop/projects/External/style_transfer/style_Image/abstraction_1.jpeg'
# print(style_path)
#
# content_image = load_img(content_path, 1024)
# style_image = load_img(style_path, 1024)
#
# style_targets = extractor(style_image)['style']
# content_targets = extractor(content_image)['content']
#
# image = tf.Variable(content_image)
# opt = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.99, epsilon=1e-2)

# # run style_transfer
# epochs = 1
# steps_per_epoch = 1
#
# step = 0
# for n in range(epochs):
#     for m in range(steps_per_epoch):
#         step += 1
#         train_step(image, loss_func=loss, optimizer=opt, style_targets=style_targets, content_targets=content_targets,
#                    style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT, tv_weight=TV_WEIGHT, extractor=extractor)
# plt.imsave("/Users/user/Desktop/projects/External/style_transfer/output/result.png", image.numpy()[0])
