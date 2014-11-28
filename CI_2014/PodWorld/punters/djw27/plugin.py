import math
from pod.pods import Sensor,gui

FORWARDS_ANGLE = math.radians(0)
LEFT_ANGLE = math.radians(90)
RIGHT_ANGLE = math.radians(-90)

class MyData:

    def __init__(self):
        pass


def equip_car(pod):


    sensors=[ Sensor(angle=FORWARDS_ANGLE,name="Forwards"),
              Sensor(angle=LEFT_ANGLE,name="Left"),
              Sensor(angle=LEFT_ANGLE/2,name="Half-Left"),
              Sensor(angle=RIGHT_ANGLE,name="Right"),
              Sensor(angle=RIGHT_ANGLE/2,name="Half-Right"),
            ]

    pod.addSensors(sensors)
    pod.col=(0,255,225)


    pod.data=MyData()
    pod.data.pram=[200,0.17,0.3359]      #   USER DATA CAN BE ANYTHING YOU WANT!
    pod.poly=[(-10,-10),(-5,20),(5,20),(10,-10)]


def controller(pod,control):

    state=pod.state
    sensors=pod.sensors

    vel=state.vel
    forwardClearance = sensors[0].val

    if vel < pod.data.pram[0]:
        control.up=pod.data.pram[1]
    else:
        control.up=0.0

    left=math.sqrt((0.25*sensors[1].val*sensors[1].val) + (sensors[2].val*sensors[2].val))
    right=math.sqrt((0.25*sensors[3].val*sensors[3].val) + (sensors[4].val*sensors[4].val))
    # ideal value here around 40?
    #print left,right

    if left < right:
        control.left=0
        control.right=pod.data.pram[2]
    else:
        control.left=pod.data.pram[2]
        control.right=0




