import numpy as np
from scipy.spatial.transform import Rotation as R

#Need to be able to input starting angle for sensor, or not.
def bend(segment_length,mount_pos,bend_angle,bend_direction):
    
    if bend_angle==0:
        bend_angle=.000001

    circle_radius = segment_length/bend_angle

    x = np.cos(np.pi-bend_angle)*circle_radius + mount_pos[0]+circle_radius
    z = np.sin(np.pi-bend_angle)*circle_radius + mount_pos[2]
    
    second_circle_radius = x-mount_pos[0]

    x1 = x * np.cos(bend_direction)
    y = np.sin(bend_direction)*second_circle_radius
     
    return np.array([x1,y,z])

def angle(pos,euler):
    r1 = R.from_euler('zyx',euler)
    return np.array(r1.apply(pos))
