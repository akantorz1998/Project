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
BUZZER = 21
VCC = 20
FAN = 5

FAN_STATE = True
GPIO.setup(FAN,GPIO.OUT)
GPIO.output(FAN,FAN_STATE)

VCC_STATE = True
GPIO.setup(VCC,GPIO.OUT)
GPIO.output(VCC,VCC_STATE)


buzzerState = False
GPIO.setup(BUZZER,GPIO.OUT)
GPIO.output(BUZZER,buzzerState)


time.sleep(1)
buzzerState = not buzzerState
FAN_STATE = not FAN_STATE
GPIO.output(BUZZER,buzzerState)
GPIO.output(FAN,FAN_STATE)