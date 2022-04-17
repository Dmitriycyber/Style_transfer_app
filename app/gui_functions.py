import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

def open_file():
    """Открываем файл для редактирования"""
    filepath = askopenfilename(
        filetypes=[("Графические файлы", "*.jpg|*jpeg"), ("Все файлы", "*.*")]
    )
    if not filepath:
        return
        txt_edit.delete("1.0", tk.END)
    else:
        return filepath