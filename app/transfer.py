import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from app.functions import load_img
from app.functions import StyleAndContentExtractor, train_step, loss, transfer

transfer(STYLE_PATH="style_transfer/misc/style_image.jpg",
         CONTENT_PATH="style_transfer/misc/content_image.jpg",
         STYLE_WEIGHT=100.0,
         CONTENT_WEIGHT=5.0,
         TV_WEIGHT=0.1,
         EPOCHS=1,
         STEPS_PER_EPOCHS=2)