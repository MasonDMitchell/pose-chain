from segment import Segment
import numpy as np
import scipy as sci
import pandas as pd

class Chain:
    def __init__(self,segment_amount):

        self.segment_amount = segment_amount
        
        #This contains all of the segment objects
        self.segments = []

        for i in range(self.segment_amount):
            self.segments.append(Segment())

        #Initial orientation and position of first segment
        self.start_pose = [0,0,0]
        self.start_rotvec = [0,0,1]

        #Length of structure between magnet and next sensor
        self.module_length = 80

        self.lines = []

    def update_chain(self):
        #Ensure initial pose & rotvec are accurate
        self.segments[0].update_pose(self.start_pose)
        
        #TODO Fix magic orientation rotvec
        self.segments[0].apply_rotvec(self.start_rotvec,[1,0,0])

        #Loop for all other segments except the first one as they are determined by previous segment
        for i in np.arange(1,self.segment_amount,1):
            #New rotvec for segment sensor
            prev_rotvec = self.segments[i-1].final_rotvec
            #Difference of position from module length
            module_difference = [self.module_length * a for a in prev_rotvec]

            #New segment position from last magnet plus module
            new_pos = [a+b for a,b in zip(self.segments[i-1].magnet_pose,module_difference)]
            #update segment and apply the most updated vector
            self.segments[i].update_pose(new_pos)
            self.segments[i].apply_rotvec(prev_rotvec,self.segments[i-1].applied_rotvec) 

        return

    #Change the bend of any segment, afterwards update_chain should be run
    def bend_segment(self,segment,bend_angle,bend_direction):
        #TODO segment is a valid number

        #Update constants
        self.segments[segment].bend_angle = bend_angle
        self.segments[segment].bend_direction = bend_direction
        
        return

    #Give a list of all of the lines between sensors & magnets, useful for visualization
    def bend_lines(self):

        self.lines = []
        #Ask segment for bend line with correct rotvec for each segment. Append to list
        for i in range(self.segment_amount):
            self.lines.append(self.segments[i].bend_line(self.segments[i].applied_rotvec))
        return self.lines
