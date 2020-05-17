import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from chain2 import Chain
import random

random.seed(a=1,version=2)

class Simulation:
    def __init__(self,chain=None):
        #Hall Effect Sensor runs at 1MHz with 8MSBs, so, we kinda only need FPS for us cause we can assume instant sensor readings.
        self.fps = 60
        if chain is None:
            self.chain = Chain()
        else:
            self.chain = chain

        self.segment_length = self.chain._segment_count

        #Units are in rad/s
        self.bend_direction_accel = [0]*self.segment_length
        self.bend_direction_vel = [0]*self.segment_length

        self.max_bend_direction_vel = 1
        self.max_bend_direction_accel = .2

        self.bend_angle_accel = [0]*self.segment_length
        self.bend_angle_vel = [0]*self.segment_length

        self.max_bend_angle_vel = 1
        self.max_bend_angle_accel = .2


    def update(self):
        for i in range(self.segment_length):
            self.random_acceleration()
            self.bend_direction_vel[i] = self.bend_direction_vel[i] + self.bend_direction_accel[i]
            self.bend_angle_vel[i] = self.bend_angle_vel[i] + self.bend_angle_accel[i]
            self.chain._segments[i].bend_angle()

    def random_acceleration(self):
        for i in range(self.segment_length):
            #If velocity is too large, move reset acceleration to 1/4 of max in needed direction
            if(self.bend_direction_vel[i] > self.max_bend_direction_vel):
                self.bend_direction_accel[i] = -self.max_bend_direction_accel/4
            if(self.bend_direction_vel[i] < -self.max_bend_direction_vel):
                self.bend_direction_accel[i] = self.max_bend_direction_accel/4

            #Add random acceleration by +- 1/10 max
            self.bend_direction_accel[i] = self.bend_direction_accel[i] + random.uniform(self.max_bend_direction_accel*.1,-self.max_bend_direction_accel*.1)

            #If acceleration is too large, decrease by 1/4 of max
            if(self.bend_direction_accel[i] > self.max_bend_direction_accel):
                self.bend_direction_accel[i] = self.bend_direction_accel - self.max_bend_direction_accel/4
            if(self.bend_direction_accel[i] < -self.max_bend_direction_accel):
                self.bend_direction_accel[i] = self.bend_direction_accel + self.max_bend_direction_accel/4

            #Bend Acceleration Randomization

            #If velocity is too large, move reset acceleration to 1/4 of max in needed direction
            if(self.bend_angle_vel[i] > self.max_bend_angle_vel):
                self.bend_angle_accel[i] = -self.max_bend_angle_accel/4
            if(self.bend_angle_vel[i] < -self.max_bend_angle_vel):
                self.bend_angle_accel[i] = self.max_bend_angle_accel/4

            #Add random acceleration by +- 1/10 max
            self.bend_angle_accel[i] = self.bend_angle_accel[i] + random.uniform(self.max_bend_angle_accel*.1,-self.max_bend_angle_accel*.1)

            #If acceleration is too large, decrease by 1/4 of max
            if(self.bend_angle_accel[i] > self.max_bend_angle_accel):
                self.bend_angle_accel[i] = self.bend_angle_accel - self.max_bend_angle_accel/4
            if(self.bend_angle_accel[i] < -self.max_bend_angle_accel):
                self.bend_angle_accel[i] = self.bend_angle_accel + self.max_bend_angle_accel/4


if __name__ == "__main__":
    from chain2 import Chain
    from segment2 import LineSegment,CircleSegment

    segment_list = []
    for i in range(10):
            segment_list.append(LineSegment(10))
            segment_list.append(CircleSegment(100,np.pi/2,np.pi/2+(.1*i)+.1))
    chain = Chain(segment_list=segment_list)
    x = Simulation(chain)
    x.update()
