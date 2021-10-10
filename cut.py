import cv2 # 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class 
# 영상이 있는 경로 

def cut():
    print("S T A R T")
    vidcap = cv2.VideoCapture('./11.mp4') 
    count = 0 
    while(vidcap.isOpened()):
        ret, image = vidcap.read()
        # 30프레임당 하나씩 이미지 추출 
        if(int(vidcap.get(1)) % 10 == 0): 
            # 추출된 이미지가 저장되는 경로 
            cv2.imwrite("./cut/frame%d.jpg" % count, image) 
            print('Saved frame%d.jpg' % count) 
            count += 1 
        
    vidcap.release()

cut()
print("E X I T C U T")
