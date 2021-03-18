# -*- coding: utf-8 -*-
#!/home/pi/Desktop/tf_pi/env/bin/python3.7
"""
Created on Tue Feb 23 15:09:46 2021

@author: nataw
"""




"""
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""
import configparser
import keras
from tensorflow.python.keras.models import load_model
config = configparser.ConfigParser()
config.read('config.ini')

model = load_model(str(config['DEFAULT']['ModelPath']))
input_size = int(config['DEFAULT']['inputsize'])


from bluetoothAPI import *
# import the necessary packages
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import time
import dlib
import numpy as np
from imutils import face_utils
import RPi.GPIO as GPIO


#import tensorflow.keras

#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True




class DrowsinessApp:
    global sess
    global graph
    def __init__(self, vs):
        #self.buzzer = buz
        self.vs = vs
        self.outputPath = None
        self.frame = None
        self.ref = None
        self.thread = None
        self.stopEvent = None
        self.thread_bluetooth = None
        self.thread_blz_send = None
        
        self.predictor = dlib.shape_predictor(str(config['DEFAULT']['shape_predictor']))
        self.detector = cv2.CascadeClassifier(str(config['DEFAULT']['CascadeClassifier']))
# initialize the root window and image panel
        self.root = tki.Tk()
        self.server_sock = bluetoothAPI()
        self.panel = None
        self.lbl = None
        self.min_ear = None
        self.max_ear = None
        self.lbl_blue = None
        self.lbl_ear = None
        self.lbl_face = None
        self.adv_EAR = 0.000
        self.lbl_mode = None
        self.EYE_AR_THRESH = float(config['DEFAULT']['Ear'])
        self.face_cascade = cv2.CascadeClassifier(str(config['DEFAULT']['CascadeClassifier']))
        #self.model = load_model('CNN_Model_newData.h5')
        # create a button, that when pressed, will take the current
# frame and save it to file
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_rowconfigure(5, weight=1)
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.BUZZER = 21
        self.VCC = 20
        self.VCC_STATE = True
        GPIO.setup(self.VCC,GPIO.OUT)
        GPIO.output(self.VCC,self.VCC_STATE)
        #self.root.grid_columnconfigure(2, weight=1)
        #self.root.grid_columnconfigure(3, weight=1)
        #self.root.grid_columnconfigure(4, weight=1)
        #self.root.grid_columnconfigure(5, weight=1)
        #self.root.grid_columnconfigure(6, weight=1)

        btn = tki.Button(self.root, text="Switch mode",
                         command=self.mode_sw)
        #btn.pack(side="bottom", fill="both", expand="yes", padx=10,
        #                 pady=10)
        btn.grid(column = 1,row = 4)
        
        self.btn_ear = tki.Button(self.root, text="EAR Calculator",
                         command=self.EAR_Adv)
        #btn.pack(side="bottom", fill="both", expand="yes", padx=10,
        #                 pady=10)
        self.btn_ear.grid(column = 1,row = 2)
        
        self.lbl_mode = self.lbl_mode = tki.Label(self.root, text="Mode : EAR")
        self.lbl_mode.config(font=("Courier", 14))
        self.lbl_mode.grid(column = 1,row = 5,columnspan = 2)
        
        
        
        self.lbl_ear = tki.Label(self.root, text="EAR = 0.000",width=15)
        self.lbl_ear.config(font=("Courier", 14))
        self.lbl_ear.grid(column = 1,row = 0)
        
        self.lbl_face = tki.Label(self.root, text="No face")
        self.lbl_face.config(font=("Courier", 12))
        self.lbl_face.grid(column = 1,row = 3)
        
        self.lblblue = tki.Label(self.root, text="Bluetooth is not connecting..")
        self.lblblue.config(font=("Courier", 14))
        self.lblblue.grid(column = 0,row = 0)
        
        self.lbl = tki.Label(self.root, text="Normally")
        self.lbl.config(font=("Courier", 20))
        #self.lbl.pack(side="right",fill="both",expand="yes",padx=10, pady=10)
        self.lbl.grid(column = 1,row = 1)
    
# start a thread that constantly pools the video sensor for # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
# set a callback to handle when the window is closed
        self.root.wm_title("Drowsiness")
        self.root.geometry('800x480')
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        self.img = None
        self.mode = 2
    def alarm(self):
        buzzerState = False
        GPIO.setup(self.BUZZER,GPIO.OUT)
        GPIO.output(self.BUZZER,buzzerState)
        time.sleep(1)
        buzzerState = not buzzerState
        GPIO.output(self.BUZZER,buzzerState)
    def alam():
        exec(open('Buzzer.py').read())
    def mode_sw(self):
        if self.mode == 1:
            self.mode = 2
            self.lbl_mode.config(text = "Mode : EAR")
        else:
            self.mode = 1
            self.lbl_mode.config(text = "Mode : CNN")
                
    def euclidean_dist(self,ptA,ptB):
    # compute and return the euclidean distance between the two
        return np.linalg.norm(ptA - ptB)
    def eye_aspect_ratio(self,eye):
        A = self.euclidean_dist(eye[1], eye[5])
        B = self.euclidean_dist(eye[2], eye[4])

        C = self.euclidean_dist(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear
    
    def blue_loop(self):
        #self.thread_blz_send = 
        call(['sudo', 'hciconfig', 'hci0', 'piscan'])
        text_connect = "Wait for connection..."
        print(text_connect)
        self.server_sock.start_server()
        #try:
        while not self.stopEvent.is_set():
            conected = self.server_sock.check_conection()
            if conected:
                self.lblblue.config(text="Bluetooth is conected")
            if not conected:
                self.lblblue.config(text="Bluetooth is not conected")
                print(text_connect)
                #if self.thread_blz_send.is_alive():
                #    self.thread_blz_send.exit()
                self.server_sock.start_server()
            if conected and self.server_sock.is_recv:
                data = self.server_sock.received()
                
                str_re = str(data)
                print("Received \"%s\" through Bluetooth"%data)
                if("end" in str_re):
                    print("Exiting for recived data")
                    #self.thread_blz_send.start()
                    self.server_sock.is_recv = False
                if "ear" in str_re:
                    recv_ear = str_re[-10:-5]
                    self.EYE_AR_THRESH = float(recv_ear)
                    print(self.EYE_AR_THRESH)
            #print("from loop")
            #print(self.server_sock.alert)
            if conected and self.server_sock.alert:
                #print("printttt")
                #self.server_sock.is_recv = False
                self.server_sock.send(self.server_sock.msg_send+"/n")
                #self.server_sock.msg_send = ""
                self.server_sock.alert = False
                #self.server_sock.is_recv = True
        #except e:
         #   print("error...")
         #   print(e)
            
    def avg(self,list_sleepy):
        sum_c = 0
        for i in list_sleepy:
            sum_c = sum_c + i
            re = sum_c / len(list_sleepy)
        print(len(list_sleepy))
        return re
    def EAR_Adv(self):
        if self.min_ear is not None and self.max_ear is not None:
            self.adv_EAR = (self.min_ear + self.max_ear)/2
            self.btn_ear.config(text="EAR : %.3f"%self.adv_EAR)
            self.EYE_AR_THRESH = self.adv_EAR
            with open('config.ini','w') as configfile:
                config['DEFAULT']['Ear'] = "%.3f"%self.adv_EAR
                config.write(configfile)
        #print(self.adv_EAR)
    def videoLoop(self):
        sleepy_cout = 0
        list_sleepy = []
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        EYE_AR_CONSEC_FRAMES = int(config['DEFAULT']['EYE_AR_CONSEC_FRAMES'])#16
        # initialize the frame counter as well as a boolean used to
        # indicate if the alarm is going off
        COUNTER = 0
        ALARM_ON = False
        sleep_state = False
# DISCLAIMER:
# I'm not a GUI developer, nor do I even pretend to be. This
# try/except statement is a pretty ugly hack to get around
# a RunTime error that Tkinter throws due to threading
        try:
            self.thread_bluetooth = threading.Thread(target=self.blue_loop,daemon=True,args=())
            self.thread_bluetooth.start()
            t1 = threading.Thread(target = self.alarm)
# keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                try:
                    
                    self.frame = self.vs.read()

                    faces = self.face_cascade.detectMultiScale(self.frame, 1.3, 5)
                    if len(faces) == 0:
                        self.lbl_face.config(text="No face")
                    else:
                        self.lbl_face.config(text="Face detected")
                        face_crop = []
                        for f in faces:
                            x, y, w, h = [ v for v in f ]
                            face_crop.append(self.frame[y:y+h, x:x+w])
                            rect = dlib.rectangle(int(x), int(y), int(x + w),
                                            int(y + h))
                        self.img = cv2.resize(face_crop[0],(input_size,input_size))
                        #pred = self.model.predict_classes(self.img.reshape(1,224,224,3)/255.)
                        #with graph.as_default():
                        #    set_session(sess)
                        if self.mode == 1:
                            pred = model.predict_classes(self.img.reshape(1,input_size,input_size,3)/255.)
                            ############################################
                            list_sleepy.append(pred)
                            if len(list_sleepy)>3:
                                list_sleepy.pop(0)
                            sleep = self.avg(list_sleepy)
                            ######################################################
                            if sleep < 0.5:
                                self.lbl.configure(text="Not Sleepy",fg="black")
                                sleepy_cout = 0
                                sleep_state = False
                            else: 
                                sleepy_cout +=1
                                if sleepy_cout > int(config['DEFAULT']['cnn_count']):
                                    self.lbl.configure(text="Sleepy",fg="red")
                                    if not sleep_state:
                                        sleep_state = True
                                        self.server_sock.alert = True
                                    if not t1.is_alive():
                                        t1 = threading.Thread(target = self.alarm)
                                        t1.start()
                                    print(self.server_sock.alert)
                        elif self.mode == 2:
                            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                            #rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                            #        minNeighbors=11, minSize=(80, 80),
                            #        flags=cv2.CASCADE_SCALE_IMAGE)
                            
                            shape = self.predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)
                            leftEye = shape[lStart:lEnd]
                            rightEye = shape[rStart:rEnd]
                            leftEAR = self.eye_aspect_ratio(leftEye)
                            rightEAR = self.eye_aspect_ratio(rightEye)
                            
                            ear = (leftEAR + rightEAR) / 2.0
                            if self.min_ear is None:
                                self.min_ear = ear
                                #print("min")
                            elif ear < self.min_ear:
                                self.min_ear = ear
                            if self.max_ear is None:
                                self.max_ear = ear
                                #print("nax")
                            elif self.min_ear > ear:
                                ear > self.max_ear
                            
                            self.lbl_ear.config(text="EAR =%.3f"%ear)
                            
                            leftEyeHull = cv2.convexHull(leftEye)
                            rightEyeHull = cv2.convexHull(rightEye)
                            
                            cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
                            cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)
                            
                            if ear < self.EYE_AR_THRESH:
                                COUNTER += 1
                                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                    #t1 = threading.Thread(target = self.alarm)
                                    if not ALARM_ON:
                                        print("alarm")
                                    self.lbl.configure(text="Sleepy",fg="red")
                                    if not sleep_state:
                                        sleep_state = True
                                        self.server_sock.alert = True
                                    if not t1.is_alive():
                                        t1 = threading.Thread(target = self.alarm)
                                        t1.start()
                                    #self.server_sock.alert = True
                            else:
                                COUNTER = 0
                                self.lbl.configure(text="Not sleepy",fg="black")
                                sleep_state = False
                                #ALARM_ON = False
                                    #print(self.server_sock.alert)
                                    
                        #####################################################
                    self.frame = cv2.resize(self.frame,(600,400))
                    image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)
                    if self.panel is None:
                        self.panel = tki.Label(image=image,background = "red",width=600)
                        self.panel.image = image
                        #self.panel.pack(side="left", padx=10, pady=10)
                        self.panel.grid(column=0, row=1,rowspan=5)
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image
                except:
                    print("camera error")
            time.sleep(2)
        except RuntimeError:
            print("[INFO] caught a RuntimeError")


    
    def takeSnapshot(self):
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))
    def onClose(self):
        print("[INFO] closing...")
        
        
        self.stopEvent.set()
        self.vs.stop()
        #self.root.destroy()
        self.root.quit()
        #call(['sudo', 'shutdown', '-h', 'now'])
        
        
        
        