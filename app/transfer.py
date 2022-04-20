import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from app.functions import load_img
from app.functions import StyleAndContentExtractor, train_step, loss, transfer

STYLE_WEIGHT = 100.0
CONTENT_WEIGHT = 5.0
TV_WEIGHT = 0.1

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
content_layers = ['block4_conv2']

extractor = StyleAndContentExtractor(style_layers=style_layers, content_layers=content_layers)
sample_image = np.ones((1, 512, 512, 3), dtype=np.float32)
style_and_content_targets = extractor(sample_image)

content_path = '/moscow-moscow-city-moskva-moskva-siti-gorod-city.jpeg'
style_path = '/Users/user/Desktop/projects/External/style_transfer/style_Image/abstraction_1.jpeg'
print(style_path)

content_image = load_img(content_path, 1024)
style_image = load_img(style_path, 1024)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)
opt = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.99, epsilon=1e-2)

# run style_transfer
epochs = 1
steps_per_epoch = 1

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image, loss_func=loss, optimizer=opt, style_targets=style_targets, content_targets=content_targets,
                   style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT, tv_weight=TV_WEIGHT, extractor=extractor)
plt.imsave("/Users/user/Desktop/projects/External/style_transfer/output/result.png", image.numpy()[0])

transfer(STYLE_PATH="style_transfer/misc/style_image.jpg",
         CONTENT_PATH="style_transfer/misc/content_image.jpg",
         STYLE_WEIGHT=100.0,
         CONTENT_WEIGHT=5.0,
         TV_WEIGHT=0.1,
         EPOCHS=1,
         STEPS_PER_EPOCHS=2)