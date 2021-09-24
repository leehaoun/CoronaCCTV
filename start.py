import hanium_detect
   
   
hanium_detect.detect(source='data/videos/testvideo.mp4', weights='weights/custom-v2.pt', conf_thres=0.55, device='0' )