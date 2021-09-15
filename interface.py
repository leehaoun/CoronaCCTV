# 2021-09-06 주석 추가, 화면 크기 조절 추가, 경보와 메세지 기능 구현

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
    # source = 0이면 웹캠 호출 , weights 저장된 가중치 불러옴 , device 라디오 버튼으로 CPU, GPU 선택
    # conf_thres 임계값을 콤보박스의 숫자만 가져와서 100으로 나눈값으로 설정 ex) 50%면 50/100 = 0.5 , hide_labels 라벨표시여부 라디오 버튼으로 선택
    win_size=win_size_combobox.get().split('x')
    hanium_detect.detect(source='0', w_width=int(win_size[0]), w_height=int(win_size[1]),
                         weights='./weights/custom.pt', device=ps_RadioVariety.get(),
                         conf_thres=float(combobox.get()[0:2])/100, hide_labels=lbon_RadioVariety.get(),
                         alarm=siren.get(), message=message.get())


# 전체 구조(1000X600의 2X2 구조)
root = Tk()
root.title("Corona CCTV")
w = 1000
h = 600
root.geometry("%dx%d"%(w, h))
root.grid_columnconfigure(0, weight=1, minsize=500)
root.grid_columnconfigure(1, weight=1, minsize=500)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

# 기본 설정
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

# 전체적 구조
frame_nw=Frame(root, relief="solid", bd=1)
frame_nw.grid(row=0, column=0, sticky="nsew")

frame_sw=Frame(root, relief="solid", bd=1)
frame_sw.grid(row=1, column=0, sticky="nsew")

frame_ne=Frame(root, relief="solid", bd=1)
frame_ne.grid(row=0, column=1, sticky="nsew")

frame_se=Frame(root,  relief="solid", bd=1)
frame_se.grid(row=1, column=1, sticky="nsew")

# 처리 프로세서
ps_label = tkinter.Label(frame_nw, text='처리프로세서')
ps_label.grid(column=0, row=0)
cpu_radio = tkinter.Radiobutton(frame_nw, text="CPU", value='cpu', variable=ps_RadioVariety)
cpu_radio.grid(column=1, row=0)
gpu_radio = tkinter.Radiobutton(frame_nw, text="GPU", value='0', variable=ps_RadioVariety)
gpu_radio.grid(column=2, row=0)

# 레이블 표시 여부
lbon_label = tkinter.Label(frame_nw, text='레이블표시')
lbon_label.grid(column=0, row=1)
lbon_radio = tkinter.Radiobutton(frame_nw, text="ON", value=False, variable=lbon_RadioVariety)
lbon_radio.grid(column=1, row=1)
lboff_radio = tkinter.Radiobutton(frame_nw, text="OFF", value=True, variable=lbon_RadioVariety)
lboff_radio.grid(column=2, row=1)

# 경보 여부
warn_label = tkinter.Label(frame_nw, text='경보')
warn_label.grid(column=0, row=2)
lamp_checkbox = tkinter.Checkbutton(frame_nw, text="경광등", variable=lamp, onvalue=True, offvalue=False)
lamp_checkbox.grid(column=1, row=2)
siren_checkbox = tkinter.Checkbutton(frame_nw, text="사이렌", variable=siren, onvalue=True, offvalue=False)
siren_checkbox.grid(column=2, row=2)
message_checkbox = tkinter.Checkbutton(frame_nw, text="메세지", variable=message, onvalue=True, offvalue=False)
message_checkbox.grid(column=3, row=2)

# 신뢰도
conf_label = tkinter.Label(frame_nw, text='신뢰도')
conf_label.grid(column=0, row=3)
values = [str(i)+"%" for i in range(30, 100, 5)]
combobox = tkinter.ttk.Combobox(frame_nw, height=3, width=10, values=values)
combobox.set('50%')
combobox.grid(column=1, row=3)

# 화면 크기 선택
win_size_label = tkinter.Label(frame_nw, text='화면 크기')
win_size_label.grid(column=0, row=4)
win_size_values = ['1280x720', '640x480', '640x384']
win_size_combobox = tkinter.ttk.Combobox(frame_nw, height=2, width=10, values=win_size_values)
win_size_combobox.set('1280x720')
win_size_combobox.grid(column=1, row=4)

# 로그 화면
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

