import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2


def select_file() -> str:
    path = filedialog.askopenfilename()
    return path

def load_image(path: str, label: tk.Label):
    image = cv2.imread(path)
    image = Image.fromarray(image)
    image = image.resize((500, 500))
    image = ImageTk.PhotoImage(image)

    label.config(image=image)
    label.image = image


def open_image(label: tk.Label):
    path = select_file()
    if (len(path) == 0):
        return

    load_image(path, label)


def create_view(window: tk.Tk, buttonsNo: int = 5) -> tk.Label:
    window.minsize(700, 500)

    right_frame = tk.Frame(master=window, bg="gray", width=500, height=500)
    left_frame = tk.Frame(master=window, bg="darkgray", width=200, height=500)

    label = tk.Label(master=right_frame, text="Load image")
    label.pack(fill=tk.BOTH, expand=True)

    right_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    left_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    for i in range(buttonsNo):
        button = tk.Button(master=left_frame, text=f"Button_{i}", command=lambda: open_image(label))
        button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    return label


if __name__ == "__main__":
    window = tk.Tk()
    label = create_view(window)
    load_image("picture.jpg", label)

    window.mainloop()
