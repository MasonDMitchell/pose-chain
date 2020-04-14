import numpy as np
import scipy as sci
from scipy.spatial.transform import Rotation as R
import math

class Segment:

    def __init__(self):

        #Length of each individual section
        self.segment_length = 300
        
        #Initial position of sensor in 3d space       
        self.sensor_pose = [0,0,0]

        #Initial position of magnet in 3d space
        self.magnet_pose = [self.sensor_pose[0],self.sensor_pose[1],self.sensor_pose[2]+self.segment_length]

        #Amount of bend from straight in radians
        self.bend_angle = 0
        #Direction of bend from +x in radians
        self.bend_direction = 0
        
        #Initial and final rotational vector
        self.init_rotvec = [0,0,1]
        self.final_rotvec = [0,0,0]

        self.circle_radius = -1
        return 

    def apply_rotvec(self,rotvec):
        
        
        rotation_vector = np.cross(self.init_rotvec,rotvec)
               
        normalize = np.linalg.norm(rotation_vector)

        if normalize == 0:
            self.update_bend()
            return self.magnet_pose
 
        rotation_vector = rotation_vector / normalize
       
        angle = np.arccos(np.dot(self.init_rotvec,rotvec)/(np.linalg.norm(self.init_rotvec)*np.linalg.norm(rotvec))) 
    
        rotation_vector = rotation_vector * angle

        rotate = R.from_rotvec(rotation_vector)

        self.update_bend()
        
        self.magnet_pose = rotate.apply(self.magnet_pose)

        self.final_rotvec = rotate.apply(self.final_rotvec)

        return self.magnet_pose

    def update_bend(self):
        
        #if radius is zero, bend is solved
        if(self.bend_angle == 0):
            self.magnet_pose = [self.sensor_pose[0],self.sensor_pose[1],self.sensor_pose[2]+self.segment_length]
            self.final_rotvet = [0,0,1]
            return self.magnet_pose,self.final_rotvec

        self.circle_radius = self.segment_length/self.bend_angle

        #Assign x & z coordinates based on circle radius
        self.magnet_pose[0] = np.cos(np.pi-self.bend_angle)*self.circle_radius + self.circle_radius
        self.magnet_pose[2] = np.sin(np.pi-self.bend_angle)*self.circle_radius

        #Circle around z axis that intersects end point
        second_circle_radius = self.magnet_pose[0]
        
        #Change x & assign y based on direction of bend
        self.magnet_pose[0] = self.magnet_pose[0] * np.cos(self.bend_direction)
        self.magnet_pose[1] = np.sin(self.bend_direction)*second_circle_radius

        #vector        
        self.final_rotvec = [np.cos(self.bend_direction)*np.sin(self.bend_angle),np.sin(self.bend_direction)*np.sin(self.bend_angle),np.cos(self.bend_angle)]

        return self.magnet_pose, self.final_rotvec
