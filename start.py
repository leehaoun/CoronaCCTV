import hanium_detect
   
   
# hanium_detect.detect(source='./data/videos/random.mp4', weights='weights/custom-v5.pt', conf_thres=0.55, device='0' ) #동영상 분석
hanium_detect.detect(source='0', weights='weights/custom-v5.pt', conf_thres=0.55, device='0' ) #웹캠 분석