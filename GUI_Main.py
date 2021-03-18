#!/home/pi/Desktop/tf_pi/env/bin/python3.7
from imutils.video import VideoStream
import cv2
from DrowsinessApp import DrowsinessApp
from bluetoothAPI import *
import time
#from Buzzer import MyBuzzer


if __name__ == '__main__':
    #vs = VideoStream(usePiCamera=True).close()
    vs = VideoStream(usePiCamera=True,resolution=(720,480),framerate=40).start()
    #time.sleep(2)
    
    
    pba = DrowsinessApp(vs)
    pba.root.mainloop()
    time.sleep(2)
    pba.root.destroy()
