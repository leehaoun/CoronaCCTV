import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import winsound
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from learning import deepcall
from threading import Thread

def sound():
    winsound.PlaySound('siren.wav', winsound.SND_FILENAME | winsound.SND_PURGE)

def check(xyxy, xyxytemp):
    if xyxy[0]-10> xyxytemp[2]:
        return False
    elif xyxy[2]+10 < xyxytemp[0]:
        return False
    elif xyxy[3]+10 < xyxytemp[1]:
        return False
    elif xyxy[1]-10 > xyxytemp[3]:
        return False
    else:
        return True

def check_final(xyxy, comp):
    if comp+10 > xyxy[0] > comp-10:
        return True
    else:
        return False


exit = [500, 500, 500, 500]
tmp = [100, 100, 100, 100]
tmo_person = [200, 200, 200, 200]
tmp_sani = [200, 300, 200, 300]
tmp_temp = [200, 400, 200, 400]
tmp_qrcd = [200, 500, 200, 500]
pcount = [200, 600, 200, 600]
siren = [200, 700, 200, 700]
MAX_PERSON = 100
count = [0, 0] #첫번째 숫자는 현재 화면에 사람이 있는지,없는지 체크 두번째 숫자는 10번연속 검출되어야 사람이 있다고 판정하기 위해서 사용
deepcall_check=[0, 0, 0] #count와 마찬가지로, 객체 검출이 특정횟수 이상 연속으로 검출되어야 사용했다고 판정하기 위해 사용

sani_x = 850
temp_x = 670
qrcd_x = 0

@torch.no_grad()
def detect(weights='yolov5s.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
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
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
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
    check_sani = [False for i in range(MAX_PERSON)]
    check_temp = [False for i in range(MAX_PERSON)]
    check_qrcd = [False for i in range(MAX_PERSON)]
    check_siren = [False for i in range(MAX_PERSON)]
    check_sani_final = [True for i in range(MAX_PERSON)]
    check_temp_final = [True for i in range(MAX_PERSON)]
    check_qrcd_final = [True for i in range(MAX_PERSON)]
    person_count = 0
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
                temp = [deepcall_check[0], deepcall_check[1], deepcall_check[2]]
                for *xyxy, conf, cls in reversed(det): #1frame
                    cv2.line(im0, (qrcd_x,0), (qrcd_x,1000), (255,0,0), 5) # blue
                    cv2.line(im0, (temp_x,0), (temp_x,1000), (0,255,0), 5) # green
                    cv2.line(im0, (sani_x,0), (sani_x,1000), (0,0,255), 5) # red
                    if cls.item() == 0.0 or cls.item() == 1.0:
                        for *xyxytmp, clstmp in reversed(det):
                            if clstmp.item() != 0.0 and clstmp.item() != 1.0 and check(xyxy, xyxytmp): #cls(손, 얼굴) clstmp(손소독,qr,온도계)
                                if cls.item() == 0.0 and clstmp.item() == 2.0:
                                    x1 = xyxytmp[0]
                                    x2 = xyxytmp[2]
                                    y1 = xyxytmp[1]
                                    y2 = xyxytmp[3]
                                    ry = (y2 - y1)/8
                                    img_trim = im0s[int(y1-ry):int(y2), int(x1):int(x2)]
                                    cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                    if len(os.listdir("./tmp/img/1")) != 0:
                                        if deepcall() == 0:
                                            plot_one_box(tmp, im0, label="check = sanitizer", color=colors(int(cls), True),
                                                         line_thickness=line_thickness)
                                            deepcall_check[0] = deepcall_check[0] + 1
                                            if deepcall_check[0] == 10:
                                                check_sani[person_count] = True
                                elif cls.item() == 1.0 and clstmp.item() == 3.0:
                                    x1 = xyxytmp[0]
                                    x2 = xyxytmp[2]
                                    y1 = xyxytmp[1]
                                    y2 = xyxytmp[3]
                                    rx = (x2 - x1) / 2
                                    img_trim = im0s[int(y1):int(y2), int(x1-rx):int(x2+rx)]
                                    cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                    if len(os.listdir("./tmp/img/1")) != 0:
                                        if deepcall() == 1:
                                            plot_one_box(tmp, im0, label="check = temperatrue",
                                                         color=colors(int(cls), True),
                                                         line_thickness=line_thickness)
                                            deepcall_check[1] = deepcall_check[1] + 1
                                            if deepcall_check[1] == 10:
                                                check_temp[person_count] = True
                                elif cls.item() == 0.0 and clstmp.item() == 4.0:
                                    x1 = xyxytmp[0]
                                    x2 = xyxytmp[2]
                                    y1 = xyxytmp[1]
                                    y2 = xyxytmp[3]
                                    rx = (x2 - x1) / 4
                                    img_trim = im0s[int(y1):int(y2), int(x1-rx):int(x2+rx)]
                                    cv2.imwrite("./tmp/img/1/out.jpg", img_trim)
                                    if len(os.listdir("./tmp/img/1")) != 0:
                                        if deepcall() == 2:
                                            plot_one_box(tmp, im0, label="check = qrcode", color=colors(int(cls), True),
                                                         line_thickness=line_thickness)
                                            deepcall_check[2] = deepcall_check[2] + 1
                                            if deepcall_check[2] == 10:
                                                check_qrcd[person_count] = True

                        check_person = True


                    if save_txt: # Write to file
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

                    if check_final(xyxy, sani_x) and not check_sani[person_count]:
                        check_sani_final[person_count] = False

                    if check_final(xyxy, temp_x) and not check_temp[person_count]:
                        check_temp_final[person_count] = False

                    if check_final(xyxy, qrcd_x) and not check_qrcd[person_count]:
                        check_qrcd_final[person_count] = False

                    if not check_sani_final[person_count] or not check_temp_final[person_count] or not check_qrcd_final[person_count]:
                        plot_one_box(siren, im0, label="WARNING", color=colors(int(200), True),
                                     line_thickness=line_thickness)
                        # 사이렌 추가 부분

                #여기부터 박스
                if check_person:
                    count[1] = 0
                    if count[0] == 0:
                        person_count = person_count + 1
                        count[0] = 1
                    plot_one_box(tmo_person, im0, label="person = O", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                else:
                    count[1] = count[1] + 1
                    if count[0] == 1 and count[1] == 10:
                        count[0] = 0

                    plot_one_box(tmo_person, im0, label="person = X", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                if check_sani[person_count]:
                    plot_one_box(tmp_sani, im0, label="sani = O", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                else:
                    plot_one_box(tmp_sani, im0, label="sani = X", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                if check_temp[person_count]:
                    plot_one_box(tmp_temp, im0, label="temp = O", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                else:
                    plot_one_box(tmp_temp, im0, label="temp = X", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                if check_qrcd[person_count]:
                    plot_one_box(tmp_qrcd, im0, label="qrcd = O", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                else:
                    plot_one_box(tmp_qrcd, im0, label="qrcd = X", color=colors(int(200), True),
                                 line_thickness=line_thickness)
                plot_one_box(pcount, im0, label=f"count = {person_count}", color=colors(int(200), True),
                             line_thickness=line_thickness)
                # 박스끝
                if temp == deepcall_check:
                    deepcall_check[0] = 0
                    deepcall_check[1] = 0
                    deepcall_check[2] = 0


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            plot_one_box(exit, im0, label="EXIT : Q", color=colors(int(200), True),
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
