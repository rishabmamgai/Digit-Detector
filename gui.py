from detector import project
import tkinter as tk
from PIL import Image, ImageGrab
import os, time


def draw():
    os.startfile(r"D:\ML\Hanwriting ML\Paint 3D - Shortcut.lnk")


def take_screenshot():

    time.sleep(2)
    image = ImageGrab.grab(bbox=(512, 356, 1012, 856)) 
    image = image.resize((28, 28))
    image.save("image.jpeg")

    prediction = project("image.jpeg")
    result.set(prediction)


def submit(): 
    name = file_entry.get()

    prediction = project(name)

    result.set(prediction)


root = tk.Tk()
root.title("Digit Detector")
root.geometry("380x100")


file_var = tk.StringVar()
b1 = tk.Button(root, text='Draw', command=draw)

file_label = tk.Label(root, text='File : ')
file_entry = tk.Entry(root, textvariable=file_var)

submit_button = tk.Button(root, text='Predict', command=submit)

screeshot = tk.Button(root, text='Take Screenshot and Predict', command=take_screenshot)


result = tk.StringVar()
result.set("")

text = tk.Label(root, text="Prediction = ")
output = tk.Label(root, textvariable=result)



file_label.grid(row=0, column=0) 
file_entry.grid(row=0 ,column=1) 

b1.grid(row=1, column=0) 
submit_button.grid(row=1, column=1) 
screeshot.grid(row=1, column=2)

text.grid(row=2, column=0)
output.grid(row=2, column=1)

root.mainloop()