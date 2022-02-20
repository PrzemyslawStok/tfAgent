import tkinter as tk
import numpy as np


def button_command(a: int, b: int):
    return lambda: print(f"button[{a}][{b}]")

if __name__ == "__main__":
    window = tk.Tk()

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

    window.mainloop()
