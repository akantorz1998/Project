import cv2
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time

vs = VideoStream(usePiCamera=True,framerate = 60).start()
time.sleep(1.0)
time_start = time.time()
fps = 0
t_fps = 0
while True:
    now_time = time.time()
    if now_time - time_start >= 1:
        t_fps = fps
        fps = 0
        time_start = time.time()
    
    frame = vs.read()
    fps +=1
    #time.sleep(1)
    #a = vs.framerate
    cv2.putText(frame, "Fps: {:.1f}".format(t_fps), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Fps: {:.1f}".format(now_time - time_start), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()