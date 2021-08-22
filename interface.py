import tkinter
from tkinter import *
import threading
from tkinter import filedialog, ttk

import hanium_detect

def btnevent():
    th = threading.Thread(target=startyolo)
    th.daemon=True
    th.start()

def startyolo():
    hanium_detect.detect(source='0', weights='./weights/custom.pt', device=ps_RadioVariety.get(),
                         conf_thres=float(combobox.get()[0:2])/100, hide_labels=lbon_RadioVariety.get(),)

root = Tk()
root.title("Corona CCTV")
w = 1000
h = 600
root.geometry("%dx%d"%(w, h))
root.grid_columnconfigure(0, weight=1, minsize=500)
root.grid_columnconfigure(1, weight=1, minsize=500)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
ps_RadioVariety = tkinter.StringVar()
ps_RadioVariety.set('0')
lbon_RadioVariety = tkinter.BooleanVar()
lbon_RadioVariety.set(False)
lamp = tkinter.BooleanVar()
lamp.set(True)
siren = tkinter.BooleanVar()
siren.set(True)
message = tkinter.BooleanVar()
message.set(True)

frame_nw=Frame(root, relief="solid", bd=1)
frame_nw.grid(row=0, column=0, sticky="nsew")

frame_sw=Frame(root, relief="solid", bd=1)
frame_sw.grid(row=1, column=0, sticky="nsew")

frame_ne=Frame(root, relief="solid", bd=1)
frame_ne.grid(row=0, column=1, sticky="nsew")

frame_se=Frame(root,  relief="solid", bd=1)
frame_se.grid(row=1, column=1, sticky="nsew")

ps_label = tkinter.Label(frame_nw, text='처리프로세서')
ps_label.grid(column=0, row=0)
cpu_radio = tkinter.Radiobutton(frame_nw, text="CPU", value='cpu', variable=ps_RadioVariety)
cpu_radio.grid(column=1, row=0)
gpu_radio = tkinter.Radiobutton(frame_nw, text="GPU", value='0', variable=ps_RadioVariety)
gpu_radio.grid(column=2, row=0)
lbon_label = tkinter.Label(frame_nw, text='레이블표시')
lbon_label.grid(column=0, row=1)
lbon_radio = tkinter.Radiobutton(frame_nw, text="ON", value=False, variable=lbon_RadioVariety)
lbon_radio.grid(column=1, row=1)
lboff_radio = tkinter.Radiobutton(frame_nw, text="OFF", value=True, variable=lbon_RadioVariety)
lboff_radio.grid(column=2, row=1)
warn_label = tkinter.Label(frame_nw, text='경보')
warn_label.grid(column=0, row=2)
lamp_checkbox = tkinter.Checkbutton(frame_nw, text="경광등", variable=lamp)
lamp_checkbox.grid(column=1, row=2)
siren_checkbox = tkinter.Checkbutton(frame_nw, text="사이렌", variable=siren)
siren_checkbox.grid(column=2, row=2)
message_checkbox = tkinter.Checkbutton(frame_nw, text="메세지", variable=message)
message_checkbox.grid(column=3, row=2)
label = tkinter.Label(frame_nw, text='신뢰도')
label.grid(column=0, row=3)
values = [str(i)+"%" for i in range(30, 100, 5)]
combobox = tkinter.ttk.Combobox(frame_nw, height=3, width=10, values=values)
combobox.set('50%')
combobox.grid(column=1, row=3)


listbox = tkinter.Listbox(frame_ne, selectmode='extend', height=0)
listbox.insert(0, "1번로그")
listbox.insert(1, "2번로그")
listbox.insert(2, "3번로그")
listbox.grid(column=0, row=0)
btn4=Button(frame_se, width=10,height=4, text="시작",command=btnevent, overrelief="solid", bd=5, bg="green")
btn4.grid(column=1, row=0)

label = tkinter.Label(frame_sw, text='sw')
label.grid(column=0, row=0)


root.mainloop()

