import usb.core
import usb.util
import os, subprocess, time



def call_siren():
    dev = usb.core.find(idVendor=0x04d8, idProduct=0xe73c)
    dev.set_configuration()
    buffer = [0x0] * 8
    # find our device    
    end_point = 0x01
    buffer[0] = 0x57  # Write mode
    buffer[1] = 0x00
    buffer[2]= 2 # LED: 0-Off, 1-On, 2-Blink
    buffer[3] = 2
    buffer[4] = 2
    buffer[5] = 0x00
    buffer[6] = 0x00
    buffer[7]= 1 # Sound: 0-Off, 1~4-On
    # write the data
    dev.write(end_point, buffer)
    time.sleep(2)
    buffer[2] = 0
    buffer[3] = 0
    buffer[4] = 0   
    buffer[7] = 0
    dev.write(end_point, buffer)
    dev.reset()