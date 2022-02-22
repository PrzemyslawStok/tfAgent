import tkinter as tk
import numpy as np


def button_command(a: int, b: int):
    return lambda: print(f"button[{a}][{b}]")


def window1(window: tk.Tk):
    xdim = 5
    ydim = 5

    for i in range(xdim):
        for j in range(ydim):
            window.columnconfigure(i, weight=1, minsize=75)
            window.rowconfigure(i, weight=1, minsize=50)

            colorRand = np.random.randint(10, 99, [3])
            frame = tk.Frame(master=window, relief=tk.RAISED, background=f"#{colorRand[0]}{colorRand[1]}{colorRand[2]}")
            frame.grid(row=i, column=j)
            label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
            button = tk.Button(master=frame, text=f"Row {i}\nColumn {j}", command=button_command(i, j))
            label.pack(padx=5, pady=5)
            button.pack(padx=5, pady=5)


def window2(window: tk.Tk, buttonsNo: int = 5) -> None:
    window.minsize(700, 500)

    right_frame = tk.Frame(master=window, bg="gray", width=500, height=500)
    left_frame = tk.Frame(master=window, bg="darkgray", width=200, height=500)

    right_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
    left_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

    for i in range(buttonsNo):
        button = tk.Button(master=left_frame, text=f"Button_{i}")
        button.pack(fill=tk.BOTH, side=tk.TOP, expand=True)


if __name__ == "__main__":
    window = tk.Tk()
    window2(window)

    window.mainloop()
