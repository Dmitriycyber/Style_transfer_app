import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
from PIL import Image

def open_file_style():
    """Choose style image"""
    filepath = askopenfilename(filetypes=[("Графические файлы", "*.jpeg"), ("Все файлы", "*.*")])
    if not os.path.exists("misc"):
        os.makedirs("misc")
    # os.mkdir("Users/user/Desktop/projects/External/style_transfer/misc")
    img = Image.open(filepath)
    os.chdir("misc")
    img.save('style_image.jpg')
    os.chdir("/Users/79038/Desktop/projects/External/style_transfer")



def open_file_content():
    """Choose content image"""
    filepath = askopenfilename(filetypes=[("Графические файлы", "*.jpeg"), ("Все файлы", "*.*")])
    if not os.path.exists("misc"):
        os.makedirs("misc")
    img = Image.open(filepath)
    os.chdir("misc")
    img.save('content_image.jpg')
    os.chdir("/Users/79038/Desktop/projects/External/style_transfer")