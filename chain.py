from segment import Segment
import numpy as np
import scipy as sci
import pandas as pd

class Chain:
    def __init__(self,segment_amount):
        self.segment_amount = segment_amount
        self.segments = []
        for i in range(self.segment_amount):
            self.segments.append(Segment())
        
        self.sensors_pose = []
        self.magnets_pose = []

        self.start_pose = [0,0,0]
        self.start_rotvec = [0,0,1]

        self.rotvecs = []

        self.module_length = 80

    def update_chain(self):

        previous_magnet_pose = self.start_pose
        previous_rotvec = self.start_rotvec
        self.sensors_pose = []
        self.magnets_pose = []
        self.rotvecs = [self.start_rotvec]

        for i in range(self.segment_amount):
            #update segment and apply the most updated vector
            self.segments[i].apply_rotvec(previous_rotvec)
            
            #attain sensor & magnet values            
            self.sensors_pose.append([previous_magnet_pose[0]+(previous_rotvec[0]*self.module_length),previous_magnet_pose[1]+(previous_rotvec[1]*self.module_length),previous_magnet_pose[2]+(previous_rotvec[2]*self.module_length)])
            self.magnets_pose.append([self.sensors_pose[i][0]+self.segments[i].magnet_pose[0],self.sensors_pose[i][1]+self.segments[i].magnet_pose[1],self.sensors_pose[i][2]+self.segments[i].magnet_pose[2]])

            self.rotvecs.append(self.segments[i].final_rotvec)

            #update values for next loop
            previous_rotvec = self.segments[i].final_rotvec
            previous_magnet_pose = self.magnets_pose[i]

        return self.sensors_pose, self.magnets_pose

    def bend_segment(self,segment,bend_angle,bend_direction):
        self.segments[segment].bend_angle = bend_angle
        self.segments[segment].bend_direction = bend_direction
        return self.segments[segment].apply_rotvec(self.rotvecs[segment])

    def bend_lines(self):
        lines = []
        for i in range(self.segment_amount): 
            x = self.segments[i].bend_line(self.rotvecs[i])
            a = []
            for j in range(len(x)):
                a.append(list(np.asarray(x[j]) + int(self.sensors_pose[i][j])))
            lines.append(a)
        return lines
