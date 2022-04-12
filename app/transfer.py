import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
import pathlib
from pathlib import Path
from tkinter.filedialog import askopenfilename
from app.style_content_extractor import StyleAndContentExtractor
from app.functions import load_img, imshow

# style_weight = 100.0
# content_weight = 5.0
# tv_weight = 0.1

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

# ttk().withdraw()
content_path = askopenfilename()
print(content_path)
# content_path = '/moscow-moscow-city-moskva-moskva-siti-gorod-city.jpeg'
style_path = '/Users/user/Desktop/projects/External/style_transfer/style_Image/abstraction_1.jpeg'
print(style_path)

content_image = load_img(content_path, 1024)
style_image = load_img(style_path, 1024)

# style_targets = extractor(style_image)['style']
# content_targets = extractor(content_image)['content']
# plt.figure()
# imshow(style_image, "Style Image")
# plt.figure()
# imshow(content_image, "Content Image")
