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
    pod.data.pram=[ 170,
                    0.01386,
                    0.1653476,
                    174.1289844,
                    369.83769,
                    0.090317,
                  ]
    pod.poly=[(-10,-10),(-5,20),(5,20),(10,-10)]


def controller(pod,control):

    state = pod.state
    sensors = pod.sensors
    data = pod.data.pram

    vel = state.vel
    forwardClearance = sensors[0].val

    if vel < data[0]:
        if forwardClearance > data[4]:
            control.up=data[1]
        else:
            control.up=data[3]
    else:
        control.up=0.0

    if forwardClearance < data[4] and vel > data[0]:
        control.down=data[5]
    else:
        control.down=0.0

    left=sensors[2].val
    right=sensors[4].val

    #print left,right

    if left < right:
        control.left=0
        control.right=data[2]
    else:
        control.left=data[2]
        control.right=0

''' STATE DATA
--  position from top left of screen (in pixels):
y
x

-- rate of change of x and y  and the velocity (to make life easy for you)
dydt
dxdt
vel


-- angle of pod from in radians anti-clockwise ?
-- 0 is pointing down    up is PI
ang        can be greater than 2pi (so it wraps around)
dangdt     rate of change of angle (spin)

-- progress in terms of trip wires crossed
pos_trips
neg_trips

   after collision with the side further information about progress  (0-1 measures progress from porevious to next trip)
seg_pos 0.4

--- Other stuff
distance_travelled     (forward and backward)
age
collide_count  (number of time we have hit a wall)
collide True/False
slip    (0-1   1 means slipping a lot)
'''

