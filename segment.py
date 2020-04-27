import numpy as np
import scipy as sci
from scipy.spatial.transform import Rotation as R
import math
import magpylib as magpy
from magpylib.source.magnet import Box

#Sensor standard deviation 2.576 +-500 G

class Segment:

    def __init__(self):

        #Length of each individual section
        self.segment_length = 300
        
        #Zeroed position of sensor and magnet for internal use
        self._sensor_pose = [0,0,0]

        self._magnet_pose = [self._sensor_pose[0],self._sensor_pose[1],self._sensor_pose[2]+self.segment_length]

        #Used when changing pose from 0
        self.pose_difference = [0,0,0]
    
        #Magnet and sensor pose used for all purposes outside the class
        self.sensor_pose = self._sensor_pose
        self.magnet_pose = self._magnet_pose

        self.magnetization = [0,0,645.1]

        self.magnet_dimension = [9.05,9.05,9.05]

        #Amount of bend from straight in radians
        self.bend_angle = 0
        #Direction of bend from +x in radians
        self.bend_direction = 0
        
        #Initial and final rotational vector
        self.init_rotvec = [0,0,1]
        self.applied_rotvec = [0,0,1]
        self.final_rotvec = [0,0,0]
        self.prev_rotvec = [0,0,1]

        self.circle_radius = -1

        #magpy sensor and magnet initialization
        self.sensor = magpy.Sensor(pos=self.sensor_pose)
        self.magnet = Box(mag=self.magnetization,dim=self.magnet_dimension,pos=self.magnet_pose)

        self.rotate = False

        self.test = [1,0,0]

        return 

    def update_pose(self,new_sensor_pose):
        #Add zeroed sensor pose plus new pose to get total pose
        self.sensor_pose = [a + b for a, b in zip(self._sensor_pose, new_sensor_pose)]
        #Add total pose to magpy sensor
        self.sensor.setPosition(self.sensor_pose)
        
        #Add zeroed magnet pose plus new pose to get total pose
        self.magnet_pose = [a + b for a, b in zip(self._magnet_pose, new_sensor_pose)]
        #Add total pose to magpy magnet
        self.magnet.setPosition(self.magnet_pose)

        #Update pose difference to new pose
        self.pose_difference = new_sensor_pose

        return

    def find_rotvec(self,rotvec):
        
        #Determine if vector given is in same direction, or exact opposite direction of current init rotvec  
        if(np.array_equal(np.around(self.init_rotvec,decimals=5),np.around(rotvec,decimals=5)) or np.array_equal(np.around(self.init_rotvec,decimals=5),-np.around(rotvec,decimals=5))):
            if abs(self.init_rotvec[1]) < .00001 and abs(self.init_rotvec[2]) < .00001:
                if self.init_rotvec[0] == 0:
                    raise ValueError('zero vector')
                else: 
                    rotation_vector = np.cross(self.init_rotvec, [0,0,1])
            rotation_vector = np.cross(self.init_rotvec, [1,0,0])
        else:
            rotation_vector = np.cross(self.init_rotvec,rotvec)

        normalize = np.linalg.norm(rotation_vector)

        if normalize == 0:
            rotation_vector = [0,0,0]
            return rotation_vector, normalize, 0

        rotation_vector = rotation_vector / normalize
       
        angle = np.arccos(np.dot(self.init_rotvec,rotvec)/(np.linalg.norm(self.init_rotvec)*np.linalg.norm(rotvec))) 
        sin = np.dot(np.cross(rotation_vector,self.init_rotvec),rotvec)/(np.linalg.norm(self.init_rotvec)*np.linalg.norm(rotvec))

        if(sin < 0):
            print("sin")
            angle = 2*math.pi-angle

        rotation_vector = rotation_vector * angle
        normalize = np.linalg.norm(rotation_vector)

        if normalize == 0:
            rotation_vector = [0,0,0]
            return rotation_vector, normalize, 0

        return rotation_vector, normalize, angle

    def apply_rotvec(self,rotvec,prev_rotvec):

        self.update_bend()

        rotation_vector, normalize, angle = self.find_rotvec(rotvec)

        self.prev_rotvec = prev_rotvec
        self.applied_rotvec = rotvec

        if normalize == 0:
            self.update_pose(self.pose_difference)
            return self.magnet_pose
 
        rotate = R.from_rotvec(rotation_vector)
        self.test=rotation_vector        
        #Rotate magnet pose, and final rotvec 
        self._magnet_pose = rotate.apply(self._magnet_pose)

        self.final_rotvec = rotate.apply(self.final_rotvec)

        #Set rotvec to have length of 1
        rotvec_normalize = np.linalg.norm(self.final_rotvec)
        
        self.final_rotvec = [a / rotvec_normalize for a in self.final_rotvec]

        #Set rotvec to have length of 1
        rotvec_normalize = np.linalg.norm(self.final_rotvec)
        
        self.final_rotvec = [a / rotvec_normalize for a in self.final_rotvec]

        #Update magnet and sensor orientation and position
 
        if(False):
            self.spin(np.pi)
        else:
            self.sensor.setOrientation(axis=rotation_vector,angle=(180*angle)/math.pi)
            if(normalize > .00001):        

                self.magnet.rotate(axis=rotation_vector,angle=(180*angle)/math.pi,anchor=self._sensor_pose)

        self.update_pose(self.pose_difference)        

        return self.magnet_pose, self.final_rotvec,self.magnet,self.sensor

    def spin(self,angle):
        rotate = R.from_rotvec((self.applied_rotvec / np.linalg.norm(self.applied_rotvec)) * angle)
        self._magnet_pose = rotate.apply(self._magnet_pose)
        self.final_rotvec = rotate.apply(self.final_rotvec)
        self.magnet.setPosition(np.array(self.magnet.position)-np.array(self.pose_difference))
        self.magnet.rotate(axis=self.applied_rotvec,angle=(180*angle)/math.pi,anchor=self._sensor_pose)
        self.magnet.setPosition(np.array(self.magnet.position)+np.array(self.pose_difference))
        self.update_pose(self.pose_difference)        
        return

    def update_bend(self):
        
        #if radius is zero, bend is solved
        if(self.bend_angle == 0):
            self._magnet_pose = [self._sensor_pose[0],self._sensor_pose[1],self._sensor_pose[2]+self.segment_length]
            self.update_pose(self.pose_difference)
            self.final_rotvec = [0,0,1]
            return self.magnet_pose,self.final_rotvec

        self.circle_radius = self.segment_length/self.bend_angle

        #Assign x & z coordinates based on circle radius
        self._magnet_pose[0] = np.cos(np.pi-self.bend_angle)*self.circle_radius + self.circle_radius
        self._magnet_pose[2] = np.sin(np.pi-self.bend_angle)*self.circle_radius

        #Circle around z axis that intersects end point
        second_circle_radius = self._magnet_pose[0]
        
        #Change x & assign y based on direction of bend
        self._magnet_pose[0] = self._magnet_pose[0] * np.cos(self.bend_direction)
        self._magnet_pose[1] = np.sin(self.bend_direction)*second_circle_radius

        self.update_pose(self.pose_difference)

        #vector        
        self.final_rotvec = [np.cos(self.bend_direction)*np.sin(self.bend_angle),np.sin(self.bend_direction)*np.sin(self.bend_angle),np.cos(self.bend_angle)]

        rotation_vector, normalize, angle = self.find_rotvec(self.final_rotvec)

        if(normalize > .00001):
            self.magnet.setOrientation(angle = (180*angle)/math.pi,axis=rotation_vector)
        rotvec_normalize = np.linalg.norm(self.final_rotvec)

        self.final_rotvec = [a / rotvec_normalize for a in self.final_rotvec]
  
        return self.magnet_pose, self.final_rotvec, self.magnet, self.sensor

    def bend_line(self,rotvec):
        x = self.bend_angle
        y = self.segment_length
        line_x = []
        line_y = []
        line_z = []
        for i in np.arange(1,10,.3):
            self.bend_angle = x/i
            self.segment_length = y/i
            self.apply_rotvec(rotvec,self.prev_rotvec)

            line_x.append(self.magnet_pose[0])
            line_y.append(self.magnet_pose[1])
            line_z.append(self.magnet_pose[2])

        line_x.append(self.sensor_pose[0])
        line_y.append(self.sensor_pose[1])
        line_z.append(self.sensor_pose[2])

        self.bend_angle = x
        self.segment_length = y 

        self.apply_rotvec(rotvec,self.prev_rotvec)

        return line_x, line_y, line_z
