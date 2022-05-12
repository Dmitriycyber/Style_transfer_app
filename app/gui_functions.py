import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
from PIL import Image

def open_file_style():
    """Choose style image"""
    filepath = askopenfilename(filetypes=[("Графические файлы", "*.jpeg"), ("Все файлы", "*.*")])
    if not os.path.exists("misc"):
        os.makedirs("misc")
    img = Image.open(filepath)
    img.save(os.getcwd() + '/misc/style_image.png')



def open_file_content():
    """Choose content image"""
    filepath = askopenfilename(filetypes=[("Графические файлы", "*.jpeg"), ("Все файлы", "*.*")])
    if not os.path.exists("misc"):
        os.makedirs("misc")
    img = Image.open(filepath)
    img.save(os.getcwd() + '/misc/content_image.png')