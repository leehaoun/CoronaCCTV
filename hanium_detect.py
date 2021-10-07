# 9월 6일 그 전에 구현된 경보, 메시지 기능 UI 연동 구현 위해서 다시 추가, UI 관련으로 detect 파라미터 추가

import argparse
from math import fabs
from re import X
import time
from pathlib import Path
import cv2
import torch
from torch.autograd.grad_mode import F
import torch.backends.cudnn as cudnn
import os
import winsound
import threading
import usb.core
import usb.util
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from learning import deepcall
from threading import Thread
from tkinter import messagebox as msg
from siren import call_siren


def check(xyxy, xyxytemp):
    if xyxy[0] - 10 > xyxytemp[2]:
        return False
    elif xyxy[2] + 10 < xyxytemp[0]:
        return False
    elif xyxy[3] + 10 < xyxytemp[1]:
        return False
    elif xyxy[1] - 10 > xyxytemp[3]:
        return False
    else:
        return True
def sani_pos_check(xyxy):
    if xyxy[0]> sani_pos[2]:
        return False
    elif xyxy[2] < sani_pos[0]:
        return False
    elif xyxy[3] < sani_pos[1]:
        return False
    elif xyxy[1] > sani_pos[3]:
        return False
    else:
        return True
def temp_pos_check(xyxy):
    if xyxy[0]> temp_pos[2]:
        return False
    elif xyxy[2] < temp_pos[0]:
        return False
    elif xyxy[3] < temp_pos[1]:
        return False
    elif xyxy[1] > temp_pos[3]:
        return False
    else:
        return True
def qr_pos_check(xyxy):
    if xyxy[0]> qr_pos[2]:
        return False
    elif xyxy[2] < qr_pos[0]:
        return False
    elif xyxy[3] < qr_pos[1]:
        return False
    elif xyxy[1] > qr_pos[3]:
        return False
    else:
        return True

def sani_pos_check(xyxy):
    if xyxy[0] > sani_pos[2]:
        return False
    elif xyxy[2] < sani_pos[0]:
        return False
    elif xyxy[3] < sani_pos[1]:
        return False
    elif xyxy[1] > sani_pos[3]:
        return False
    else:
        return True


def temp_pos_check(xyxy):
    if xyxy[0] > temp_pos[2]:
        return False
    elif xyxy[2] < temp_pos[0]:
        return False
    elif xyxy[3] < temp_pos[1]:
        return False
    elif xyxy[1] > temp_pos[3]:
        return False
    else:
        return True


def qr_pos_check(xyxy):
    if xyxy[0] > qr_pos[2]:
        return False
    elif xyxy[2] < qr_pos[0]:
        return False
    elif xyxy[3] < qr_pos[1]:
        return False
    elif xyxy[1] > qr_pos[3]:
        return False
    else:
        return True


def check_Cross(x, comp):
    if comp + 10 > x > comp - 10:
        return True
    else:
        return False

# --------------텍스트박스 위치--------------------#
center = [960, 100, 960, 100]
tmp = [100, 100, 100, 100]
tmp_mask = [1600, 50, 1600, 50]
tmp_sani = [1600, 100, 1600, 100]
tmp_temp = [1600, 150, 1600, 150]
tmp_qrcd = [1600, 200, 1600, 200]
exit = [1600, 250, 1600, 250]
siren = [200, 700, 200, 700]
sani_pos = [0,0,0,0]
temp_pos = [0,0,0,0]
qr_pos = [0,0,0,0]
# ---------------텍스트박스 위치-------------------#
deepcall_check = [0, 0, 0, 0]  # 객체 검출이 특정횟수 이상 연속으로 검출되어야 사용했다고 판정하기 위해 사용
detected_sani_count = [0] # sani의 검출 횟수를 담는 변수
detected_temp_count = [0] # temp의 검출 횟수를 담는 변수
detected_qr_count = [0] # qr의 검출 횟수를 담는 변수
detected_mask_count = [0] # mask의 검출 횟수를 담는 변수

@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           w_width=1280,
           w_height=720,
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           alarm=True,
           message=True,
           mod=0
           ):
    sani_pos = [1550, 380, 1650, 480]
    temp_pos = [950, 270, 1050, 370]
    qr_pos = [420, 260, 520, 360]
    mode_check = []
    if mod == 0:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(2.0)
        mode_check.append(3.0)
        mode_check.append(4.0)
        mode_check.append(5.0)
    elif mod == 1:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(3.0)
        mode_check.append(4.0)
        mode_check.append(5.0)
    elif mod == 2:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(2.0)
        mode_check.append(4.0)
        mode_check.append(5.0)
    elif mod == 3:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(2.0)
        mode_check.append(3.0)
        mode_check.append(5.0)
    elif mod == 4:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(2.0)
        mode_check.append(5.0)
    elif mod == 5:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(3.0)
        mode_check.append(5.0)
    elif mod == 6:
        mode_check.append(0.0)
        mode_check.append(1.0)
        mode_check.append(4.0)
        mode_check.append(5.0)



    check_sani = False # BBOX 겹침이 발생했을 때, sani의 bbox를 custom-Layer로 보내서, 손소독제를 짜는 상황의 sani인지 체크하는 변수
    check_temp = False # BBOX 겹침이 발생했을 때, temp의 bbox를 custom-Layer로 보내서, 열을 재고있는 temp인지 체크하는 변수
    check_qrcd = False # BBOX 겹침이 발생했을 때, qrcd의 bbox를 custom-Layer로 보내서, qr검사를 하고 있는 qr인지 체크하는 변수
    start_x = 0 # 사람이 입장하고, 마스크착용 검사를 하는 x좌표
    sani_x_start = 0
    temp_x_start = 0
    qrcd_x_start = 0
    sani_x_end = 0 # sani_check의 값에 따라 경보를 울릴것인지, 말 것인지 결정하는 위치를 담음
    temp_x_end = 0 # temp_check의 값에 따라 경보를 울릴것인지, 말 것인지 결정하는 위치를 담음
    qrcd_x_end = 0 # qrcd_check의 값에 따라 경보를 울릴것인지, 말 것인지 결정하는 위치를 담음
    key = False# False = 설정모드, True = 검출모드
    init_check=[0,0,0] #3객체가 적당한 위치에 배치되었는지 확인하는 용도, [1,1,1]이 저장된다면 key를 true로 바꾸고 검출모드 시작
    init_check = [0, 0, 0]  # 3객체가 적당한 위치에 배치되었는지 확인하는 용도, [1,1,1]이 저장된다면 key를 true로 바꾸고 검출모드 시작
    sani_lock = [False, False] # sani의 검출이 sani_x 주변에서 딱 1번만 실행하도록 하는 용도
    temp_lock = [False, False] # temp의 검출이 temp_x 주변에서 딱 1번만 실행하도록 하는 용도
    qr_lock = [False, False] # qr의 검출이 qr 주변에서 딱 1번만 실행하도록 하는 용도
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    def sani_lock_free():
            sani_lock[0] = False
            sani_lock[1] = False
    def temp_lock_free():
            temp_lock[0] = False
            temp_lock[1] = False
    def qr_lock_free():
            qr_lock[0] = False
            qr_lock[1] = False
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, w_width=w_width, w_height=w_height)
    else:
        view_img = check_imshow()
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for dirpath, dirnames, filenames in os.walk('runs/detect/sanitizer'):
        # Remove regular files, ignore directories
        for filename in filenames:
            os.unlink(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk('runs/detect/temperature'):
        # Remove regular files, ignore directories
        for filename in filenames:
            os.unlink(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk('runs/detect/qr'):
        # Remove regular files, ignore directories
        for filename in filenames:
            os.unlink(os.path.join(dirpath, filename))
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                check_person = False
                if init_check != [1, 1, 1]:
                    init_check = [0, 0, 0]
                else:
                    key = True
                
                for *xyxy, conf, cls in reversed(det):  # reversed(det) = 1box, cls 0.0 = head , 1.0 = hands, 2.0 = sanitizer, 3.0 = temperature, 4.0 = qrcd 
                    if cls.item() in mode_check:
                        if cls.item() == 2.0 and sani_pos_check(xyxy):
                            init_check[0] = 1
                        if cls.item() == 3.0 and temp_pos_check(xyxy):
                            init_check[1] = 1
                        if cls.item() == 4.0 and qr_pos_check(xyxy):
                            init_check[2] = 1
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        # 여기부터 우리가 추가한 코드이고, 3객체를 올바른 위치에 배치시키면 key를 True로 바꾸면서 검출모드가 시작된다.
                        if key:
                            plot_one_box(center, im0, label="ON DETECTING", color=colors(int(0), True),
                                        line_thickness=line_thickness)
                            if cls.item() == 2.0:
                                sani_x_start = int(xyxy[0] + 150)
                                sani_x_end = int(xyxy[0] - 150)
                            if cls.item() == 3.0:
                                temp_x_start = int(xyxy[0] + 150)    
                                temp_x_end = int(xyxy[0] - 150)
                            if cls.item() == 4.0:
                                qrcd_x_start = int(xyxy[0] + 150)
                                qrcd_x_end = int(xyxy[0] - 150)
                            start_x = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 100

                            cv2.line(im0, (start_x, 0), (start_x, 1000), (255, 255, 255), 1)
                            cv2.line(im0, (sani_x_end, 0), (sani_x_end, 1000), (0, 0, 255), 1)  # red   
                            cv2.line(im0, (temp_x_end, 0), (temp_x_end, 1000), (0, 255, 0), 1)  # green                                                     
                            cv2.line(im0, (qrcd_x_end, 0), (qrcd_x_end, 1000), (255, 0, 0), 1)  # blue                           
                            if(sani_lock[0] == False):
                                cv2.line(im0, (sani_x_start, 0), (sani_x_start, 1000), (0, 0, 255), 1) # red
                            
                            if(temp_lock[0] == False):
                                cv2.line(im0, (temp_x_start, 0), (temp_x_start, 1000), (0, 255, 0), 1)  # green 

                            if(qr_lock[0] == False):
                               cv2.line(im0, (qrcd_x_start, 0), (qrcd_x_start, 1000), (255, 0, 0), 1)  # blue
                               
                            
                            if cls.item() == 0.0 or cls.item() == 1.0:
                                # check mask
                                if cls.item() == 1.0:
                                    if start_x > xyxy[0] > start_x - 50 and 5.0 in mode_check:
                                        x1 = xyxy[0]
                                        x2 = xyxy[2]
                                        y1 = xyxy[1]
                                        y2 = xyxy[3]
                                        img_trim = im0s[int(y1):int(y2), int(x1):int(x2)]
                                        cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                        if deepcall() == 7:
                                            plot_one_box(tmp, im0, label="check = not mask", color=colors(int(cls), True),
                                                        line_thickness=line_thickness)
                                            deepcall_check[3] = deepcall_check[3] + 1
                                            if deepcall_check[3] == 5:
                                                detected_mask_count[0] = detected_mask_count[0] + 1 
                                                plot_one_box(siren, im0, label="Not Mask!!!", color=colors(int(200), True),
                                                            line_thickness=line_thickness)                                            
                                                if alarm:
                                                    th1 = Thread(target=call_siren)
                                                    th1.start()
                                                deepcall_check[3] = 0    
                                        else:
                                            plot_one_box(tmp, im0, label="check = mask", color=colors(int(cls), True),
                                                        line_thickness=line_thickness)

                                for *xyxytmp, clstmp in reversed(det):
                                    if clstmp.item() in mode_check:
                                        if clstmp.item() != 0.0 and clstmp.item() != 1.0 and check(xyxy,xyxytmp):  # cls(손, 얼굴) clstmp(손소독,qr,온도계)
                                            if cls.item() == 0.0 and clstmp.item() == 2.0:
                                                x1 = xyxytmp[0]
                                                x2 = xyxytmp[2]
                                                y1 = xyxytmp[1]
                                                y2 = xyxytmp[3]
                                                ry = (y2 - y1) / 8
                                                img_trim = im0s[int(y1 - ry):int(y2), int(x1):int(x2)]
                                                cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                                if len(os.listdir("./tmp/img/1")) != 0:
                                                    if deepcall() == 0:
                                                        plot_one_box(tmp, im0, label="check = sanitizer",
                                                                    color=colors(int(cls), True),
                                                                    line_thickness=line_thickness)
                                                        deepcall_check[0] = deepcall_check[0] + 1
                                                        if deepcall_check[0] == 5:
                                                            check_sani = True
                                                            deepcall_check[0] = 0
                                            elif cls.item() == 1.0 and clstmp.item() == 3.0:
                                                x1 = xyxytmp[0]
                                                x2 = xyxytmp[2]
                                                y1 = xyxytmp[1]
                                                y2 = xyxytmp[3]
                                                rx = (x2 - x1) / 2
                                                img_trim = im0s[int(y1):int(y2), int(x1 - rx):int(x2 + rx)]
                                                cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                                if len(os.listdir("./tmp/img/1")) != 0:
                                                    if deepcall() == 2:
                                                        plot_one_box(tmp, im0, label="check = temperatrue",
                                                                    color=colors(int(cls), True),
                                                                    line_thickness=line_thickness)
                                                        deepcall_check[1] = deepcall_check[1] + 1
                                                        if deepcall_check[1] == 5:
                                                            check_temp = True
                                                            deepcall_check[1] = 0
                                            elif cls.item() == 0.0 and clstmp.item() == 4.0:
                                                x1 = xyxytmp[0]
                                                x2 = xyxytmp[2]
                                                y1 = xyxytmp[1]
                                                y2 = xyxytmp[3]
                                                rx = (x2 - x1) / 2
                                                img_trim = im0s[int(y1):int(y2), int(x1 - rx):int(x2 + rx)]
                                                cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                                if len(os.listdir("./tmp/img/1")) != 0:
                                                    if deepcall() == 4:
                                                        plot_one_box(tmp, im0, label="check = qrcode",
                                                                    color=colors(int(cls), True),
                                                                    line_thickness=line_thickness)
                                                        deepcall_check[2] = deepcall_check[2] + 1
                                                        if deepcall_check[2] == 5:
                                                            check_qrcd = True
                                                            deepcall_check[2] = 0

                            if cls.item() == 1.0:
                                if check_Cross(xyxy[0], sani_x_start) and sani_lock[0] == False:
                                    check_sani = False
                                    sani_lock[0] = True
                                    threading.Timer(20, sani_lock_free).start()
                                if check_Cross(xyxy[0], temp_x_start) and temp_lock[0] == False:
                                    check_temp = False
                                    temp_lock[0] = True
                                    threading.Timer(20, temp_lock_free).start()
                                if check_Cross(xyxy[0], qrcd_x_start) and qr_lock[0] == False:      
                                    check_qrcd= False
                                    qr_lock[0] = True
                                    threading.Timer(20, qr_lock_free).start()

                                if check_Cross(xyxy[2], sani_x_end) and 2.0 in mode_check:
                                    sani_lock[0] = False
                                    sani_lock[1] = False
                                if check_Cross(xyxy[2], temp_x_end) and 3.0 in mode_check:
                                    temp_lock[0] = False
                                    temp_lock[1] = False
                                if check_Cross(xyxy[2], qrcd_x_end) and 4.0 in mode_check:
                                    qr_lock[0] = False
                                    qr_lock[1] = False
                                if check_Cross(xyxy[0], sani_x_end) and 2.0 in mode_check:
                                    if check_sani == False and sani_lock[1] == False:
                                        plot_one_box(siren, im0, label="Not Sani!!!", color=colors(int(200), True),
                                                    line_thickness=line_thickness)
                                        detected_sani_count[0] = detected_sani_count[0] + 1              
                                        if alarm:
                                            th1 = Thread(target=call_siren)
                                            th1.start()
                                        sani_lock[1] = True
                                        threading.Timer(30, sani_lock_free).start()    

                                if check_Cross(xyxy[0], temp_x_end) and 3.0 in mode_check:
                                    if check_temp == False and temp_lock[1] == False:
                                        plot_one_box(siren, im0, label="Not temp!!!", color=colors(int(200), True),
                                                    line_thickness=line_thickness)
                                        detected_temp_count[0] = detected_temp_count[0] + 1             
                                        if alarm:
                                            th1 = Thread(target=call_siren)
                                            th1.start()
                                        temp_lock[1] = True
                                        threading.Timer(30, temp_lock_free).start()
                                
                                if check_Cross(xyxy[0], qrcd_x_end) and 4.0 in mode_check:
                                    if check_qrcd == False and qr_lock[1] == False:
                                        plot_one_box(siren, im0, label="Not qrcd!!!", color=colors(int(200), True),
                                                    line_thickness=line_thickness)
                                        detected_qr_count[0] = detected_qr_count[0] + 1
                                        if alarm:
                                            th1 = Thread(target=call_siren)
                                            th1.start()
                                        qr_lock[1] = True
                                        threading.Timer(30, qr_lock_free).start()
                    if key: # key=true로 설정된 이후에 보여지는 것들입니다.
                        if 5.0 in mode_check:
                            plot_one_box(tmp_mask, im0, label="mask_detect = %d" % detected_mask_count[0], color=colors(int(200), True),
                                                line_thickness=line_thickness)
                        if 2.0 in mode_check:
                            plot_one_box(tmp_sani, im0, label="sani_detect = %d" % detected_sani_count[0], color=colors(int(200), True),
                                            line_thickness=line_thickness)
                        if 3.0 in mode_check:
                            plot_one_box(tmp_temp, im0, label="temp_detect = %d" % detected_temp_count[0], color=colors(int(200), True),
                                            line_thickness=line_thickness)
                        if 4.0 in mode_check:
                            plot_one_box(tmp_qrcd, im0, label="qrcd_detect = %d" % detected_qr_count[0], color=colors(int(200), True),
                                            line_thickness=line_thickness)
                    #여기까지가 우리가 수정한 부분입니다.

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            plot_one_box(exit, im0, label="EXIT : Q", color=colors(int(200), True),
                         line_thickness=line_thickness)
            if key == False:
                plot_one_box(center, im0, label="SETTING.....", color=colors(int(0), True),
                                line_thickness=line_thickness)
                if 2.0 in mode_check:
                  plot_one_box(sani_pos, im0, label="Place Sani", color=colors(0, True),
                                  line_thickness=line_thickness)
                if 3.0 in mode_check:
                    plot_one_box(temp_pos, im0, label="Place Temp", color=colors(127, True),
                                    line_thickness=line_thickness)
                if 4.0 in mode_check:
                    plot_one_box(qr_pos, im0, label="Place QR", color=colors(255, True),
                                    line_thickness=line_thickness)             

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))
