import tkinter
import tkinter.ttk
from tkinter import *
import threading
from tkinter import filedialog

import hanium_detect

def btnevent():
    th = threading.Thread(target=startyolo)
    th.daemon=True
    th.start()

def startyolo():
    hanium_detect.detect(source='0', weights='./weights/custom.pt', device=RadioVariety_1.get(),conf_thres=float(combobox.get()[0:2])/100)

root = Tk()
root.title("Corona CCTV")
w = 200
h = 200
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
x = (sw-w)/2
y = (sh-h)/2
root.geometry("%dx%d+%d+%d"%(w,h,x,y))

RadioVariety_1 = tkinter.StringVar()
RadioVariety_1.set('0')

radio1 = tkinter.Radiobutton(root, text="CPU", value='cpu', variable=RadioVariety_1)
radio1.grid(column=0, row=0)
radio2 = tkinter.Radiobutton(root, text="GPU", value='0', variable=RadioVariety_1)
radio2.grid(column=1, row=0)

label = tkinter.Label(root, text='신뢰도')
label.grid(column=0, row=1)

values = [str(i)+"%" for i in range(30, 100, 5)]
combobox = tkinter.ttk.Combobox(root, height=3, width=10, values=values)
combobox.set('50%')
combobox.grid(column=1, row=1)

btn4=Button(root, width=10,height=4, text="시작",command=btnevent, overrelief="solid", bd=5, bg="yellow")
btn4.grid(column=1, row=2, pady=40)


root.mainloop()

