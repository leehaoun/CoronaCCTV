import hanium_detect
   
   
hanium_detect.detect(source='./data/videos/random.mp4', weights='weights/custom-v6.pt', conf_thres=0.4, device='0', mod=0) #동영상 분석 mod 0 = 전체, 1 = sani빼고, 2 = temp빼고, 3 = qr빼고, 4 = sani만, 5 = temp만, 6 = qr만
# hanium_detect.detect(source='0', weights='weights/custom-v6.pt', conf_thres=0.55, device='0', w_width=1920, w_height=1080) #웹캠 분석