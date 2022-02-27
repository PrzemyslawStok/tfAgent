import tkinter as tk
import threading
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2

video_thread: threading.Thread
video_condition: threading.Condition = threading.Condition()
video_path: str = ""

video_capture: cv2.VideoCapture

video_paused = True

def load_video(path: str):
    global video_capture
    if (len(path) == 0):
        video_capture = None
        return

    video_capture = cv2.VideoCapture(path)


def select_file(label: tk.Label) -> str:
    path = filedialog.askopenfilename()
    load_video(path)
    play_frame(label)
    return path


def create_view(window: tk.Tk) -> tk.Label:
    window.minsize(700, 500)

    right_frame = tk.Frame(master=window, bg="gray", width=500, height=500)
    left_frame = tk.Frame(master=window, bg="darkgray", width=200, height=500)

    label = tk.Label(master=right_frame, text="Load image")
    label.pack(fill=tk.BOTH, expand=True)

    right_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    left_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    button = tk.Button(master=left_frame, text=f"Load file", command=lambda: select_file(label))
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    button = tk.Button(master=left_frame, text=f"Start", command=start)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)
    button = tk.Button(master=left_frame, text=f"Pause", command=pause)
    button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    return label


def process_frame(frame: np.ndarray) -> np.ndarray:
    return frame

def play_frame(label: tk.Label) -> bool:
    if (video_capture.isOpened() == False):
        return False

    ret, frame = video_capture.read()
    if ret:
        frame = process_frame(frame)
        image = Image.fromarray(frame)
        image = image.resize((500, 500))
        image = ImageTk.PhotoImage(image)

        label.config(image=image)
        label.image = image

        return True
    else:
        return False


def play_video(label: tk.Label):
    video_condition.acquire()
    video_condition.wait()

    while (True):
        if (video_paused):
            video_condition.wait()
        if not play_frame(label):
            break

def start():
    global video_paused

    if not video_paused:
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

    video_thread = threading.Thread(target=play_video, args=(label,))
    video_thread.daemon = True
    video_thread.start()

    window.mainloop()