import usb.core
import usb.util
import os, subprocess, time



def call_siren(light, sound):
    if light:
        set_light = 2
    else:
        set_light = 0

    if sound:
        set_sound = 1
    else:
        set_sound = 0
    dev = usb.core.find(idVendor=0x04d8, idProduct=0xe73c)
    dev.set_configuration()
    buffer = [0x0] * 8
    # find our device    
    end_point = 0x01
    buffer[0] = 0x57  # Write mode
    buffer[1] = 0x00
    buffer[2]= set_light # LED: 0-Off, 1-On, 2-Blink
    buffer[3] = set_light
    buffer[4] = set_light
    buffer[5] = 0x00
    buffer[6] = 0x00
    buffer[7]= set_sound # Sound: 0-Off, 1~4-On
    # write the data
    dev.write(end_point, buffer)
    time.sleep(2)
    buffer[2] = 0 #빨간등
    buffer[3] = 0 #노란등
    buffer[4] = 0 #초록색등 
    buffer[7] = 0 #사운드
    dev.write(end_point, buffer)
    dev.reset()