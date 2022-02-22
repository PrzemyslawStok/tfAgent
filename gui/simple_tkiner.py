import tkinter as tk
import threading
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2

video_thread: threading.Thread
video_condition: threading.Condition = threading.Condition()
video_path: str = ""

video_paused = False

def select_file() -> str:
    global video_path
    path = filedialog.askopenfilename()
    video_path = path
    return path

def create_view(window: tk.Tk) -> tk.Label:
    window.minsize(700, 500)

    right_frame = tk.Frame(master=window, bg="gray", width=500, height=500)
    left_frame = tk.Frame(master=window, bg="darkgray", width=200, height=500)

    label = tk.Label(master=right_frame, text="Load image")
    label.pack(fill=tk.BOTH, expand=True)

    right_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    left_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    button = tk.Button(master=left_frame, text=f"Load file", command=select_file)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    button = tk.Button(master=left_frame, text=f"Start", command=start)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    button = tk.Button(master=left_frame, text=f"Pause", command=pause)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    return label


def play_video(path: str, label: tk.Label):
    global video_path

    video_condition.acquire()
    video_condition.wait()

    if (len(video_path) == 0):
        return

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False):
        return

    while (cap.isOpened()):
        if (video_paused):
            video_condition.wait()

        ret, frame = cap.read()
        if ret == True:
            image = Image.fromarray(frame)
            image = image.resize((500, 500))
            image = ImageTk.PhotoImage(image)

            label.config(image=image)
            label.image = image
        else:
            break


def start():
    global video_paused
    if (len(video_path) == 0):
        return

    video_paused = False

    video_condition.acquire()
    video_condition.notify()
    video_condition.release()

def pause():
    global video_paused
    video_paused = True


if __name__ == "__main__":
    window = tk.Tk()
    label = create_view(window)

    video_thread = threading.Thread(target=play_video, args=(video_path, label))
    video_thread.daemon = True
    video_thread.start()

    window.mainloop()
