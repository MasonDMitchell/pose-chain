import numpy as np
from scipy.spatial.transform import Rotation as R

#Need to be able to input starting angle for sensor, or not.
#mount pos is starting (x,y,z) of the system
#bend angle is amount of bend from straight upwards, in radians
#bend direction is direction of bend from +x, in radians

def bend(segment_length,mount_pos,bend_angle,bend_direction):
    
    #Exception for when there is no bend, as otherwise division by zero happens.    
    if bend_angle==0:
        bend_angle=.000001

    #Radius of circle that cylinder follows
    circle_radius = segment_length/bend_angle
    
    #Assign x & z coordinates based on circle radius & initial pose
    x = np.cos(np.pi-bend_angle)*circle_radius + mount_pos[0]+circle_radius
    z = np.sin(np.pi-bend_angle)*circle_radius + mount_pos[2]

    #Circle around z axis that intersects end point    
    second_circle_radius = x-mount_pos[0]

    #Change x & assign y based on direction of bend
    x1 = x * np.cos(bend_direction)
    y = np.sin(bend_direction)*second_circle_radius+mount_pos[1]
     
    return np.array([x1,y,z])

#Applying rotations to system to attempt to change z axis up assumption
def angle(pos,euler):
    r1 = R.from_euler('zyx',euler)
    return np.array(r1.apply(pos))
