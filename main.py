import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tkinter as tk
from app.gui_functions import open_file_style, open_file_content
import os
from tkinter.filedialog import askopenfilename
from app.functions import load_img
from app.functions import transfer


window = tk.Tk()
window.title("Style Transfer")
window.rowconfigure(0, minsize=500, weight=1)
window.columnconfigure(1, minsize=500, weight=1)

txt_edit_style = tk.Text(window)
fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_open_style = tk.Button(fr_buttons, text="Choose style image", command=open_file_style)
btn_open_content = tk.Button(fr_buttons, text="Choose content image", command=open_file_content)
btn_transfer = tk.Button(fr_buttons, text="Choose content image", command=transfer)
###
btn_open_style.grid(row=0, column=0, sticky="ew", padx=7, pady=7)
btn_open_content.grid(row=3, column=0, sticky="ew", padx=7, pady=7)
btn_transfer.grid(row=5, column=0, sticky="ew", padx=7, pady=7)
fr_buttons.grid(row=0, column=0, sticky="ns")


# label = tk.Label(text="Hello, Tkinter", fg="black", width=30, height=30)
# label.pack()
window.mainloop()
