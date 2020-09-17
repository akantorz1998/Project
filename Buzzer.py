import time
import RPi.GPIO as GPIO

#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)
#BUZZER = 23
#buzzerState = False
#GPIO.setup(BUZZER,GPIO.OUT)



#GPIO.output(BUZZER,buzzerState)
#time.sleep(1)
#buzzerState = not buzzerState
#GPIO.output(BUZZER,buzzerState)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
BUZZER = 23
buzzerState = False
GPIO.setup(BUZZER,GPIO.OUT)
GPIO.output(BUZZER,buzzerState)
time.sleep(1)
buzzerState = not buzzerState
GPIO.output(BUZZER,buzzerState)