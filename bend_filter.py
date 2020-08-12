import numpy as np
import magpylib as magpy
import scipy as sci
import time
from segment2 import LineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R

class Filter:
    def __init__(self,N,sensor_data,timestep):

        self.timestep = timestep
        self.print_times = False

        #Constants
        self.mag = np.array([0,0,-575.4])
        self.dim = np.array([6.35,6.35,6.35])
        self.P = 1000
        self.MAG = np.array(np.tile(self.mag,(N**2*self.P,1)))
        self.DIM = np.array(np.tile(self.dim,(N**2*self.P,1)))
        self.bend_angle_noise = 4
        self.bend_direction_noise = 0
        self.resample_percent = 60
        self.segment_length = 34.68444

        #Inputs
        self.N = N
        self.sensor_data = np.array(sensor_data)

    def compute_pose(self):
        #This can be WAY more efficient if I use the chain class instead 
        for i in range(self.P):
            particle = self.particles[i] #Select specific particle
            #Need to convert particle[0] and particle[1] to radian
            arc = CircleSegment(self.segment_length,np.radians(particle[0])%(np.pi*2),np.radians(particle[1])%(np.pi*2)) #Generate arc segment
            magnet_pose = arc.GetPoints(np.array([1]))[0] #get pose at end of arc
            magnet_axis = arc.GetOrientations(np.array([1])).as_rotvec()[0] #get orientation of end of arc as rotvec
            magnet_angle = np.linalg.norm(magnet_axis)*180/np.pi #convert rotvec length to degrees
            magnet_axis = magnet_axis/np.linalg.norm(magnet_axis) #normalize magnet axis
           
            #Update particle
            particle[2] = magnet_pose
            particle[3] = magnet_angle
            particle[4] = magnet_axis

            self.particles[i] = particle
            #Note: Not doing any sensor pose updating yet, will be done with Chain class

    def compute_flux(self): 

        y = time.time()

        POSm = self.particles[:,2] #Magnet positions
        POSm = np.repeat(POSm,self.N,0)
        POSm = np.concatenate(POSm)
        POSm = np.reshape(POSm,(self.P,3))
        
        ANG = self.particles[:,3] #Magnet angles
        ANG = np.repeat(ANG,self.N,0)
        ANG = ANG.astype("float64")

        AXIS = self.particles[:,4] #Magnet axis
        AXIS = np.repeat(AXIS,self.N,0)
        AXIS = np.concatenate(AXIS)
        AXIS = np.reshape(AXIS,(self.P,3))

        POSo = self.particles[:,5] #Sensor positions
        POSo = np.repeat(POSo,self.N,0)
        POSo = np.concatenate(POSo)
        POSo = np.reshape(POSo,(self.P,3))
        x = time.time()
        
        self.Bv = magpy.vector.getBv_magnet('box',MAG=self.MAG,DIM=self.DIM,POSo=POSo,POSm=POSm,ANG=[ANG],AX=[AXIS],ANCH=[POSm])

        x = time.time()-x
    
        self.Bv = np.reshape(self.Bv,(self.N*self.P,self.N,3))
        self.Bv = np.sum(self.Bv,1)

        ANGo = self.particles[:,6].astype("float64")
        ANGo = np.radians(ANGo)
        ANGo = np.repeat(ANGo,3,0)
        ANGo = ANGo.flatten()

        AXISo = self.particles[:,7]
        AXISo = np.concatenate(AXISo)
        AXISo = AXISo.flatten()

        rotvecs = np.multiply(ANGo,AXISo)
        rotvecs = np.reshape(rotvecs,(self.N*self.P,3))

        r = R.from_rotvec(rotvecs)
        
        self.Bv = r.apply(self.Bv,inverse=True)
        self.Bv = np.reshape(self.Bv,(self.P,self.N,3))

        y = time.time()-y
        print(self.Bv[0])
        if self.print_times == True:
            print("BV processing took: " + str(x))
            print("Non-BV compute processing took: " + str(y-x))
            print("Compute processing took: " + str(y))
        return self.Bv

    def reweigh(self):
        x = time.time() 
        print(self.sensor_data[self.timestep])
        error = np.subtract(self.sensor_data[self.timestep],self.Bv) #3D Difference between sensor and particle data
        error = np.linalg.norm(error,axis=2) #Length of 3D difference
        error = -(error*error)
        error = np.exp(error)
        error = np.sum(error,axis=1)
        error = np.reshape(error,(self.P))

        error = np.divide(error,np.sum(error))
        self.particles[:,8] = error
        if self.print_times == True:
            print("Reweigh processing took: " + str(time.time()-x))

        return error

    def predict(self):
        x = time.time()

        self.pos = np.sum(np.multiply(self.particles[:,8],self.particles[:,2]))
        #self.angle = np.sum(np.multiply(self.particles[:,6],self.particles[:,4]))
        #self.axis = np.sum(np.multiply(self.particles[:,6],self.particles[:,5]))

        #self.data = np.sum(np.multiply(np.reshape(np.repeat(self.particles[:,6],3),(self.P,3)),np.reshape(self.Bv,(self.P,3))),axis=0)
        
        if self.print_times == True:
            print("Predict processing took: " + str(time.time()-x))

    def update(self):
        
        x = time.time()

        bend_angle = self.particles[:,0]+np.random.normal(0,self.bend_angle_noise,(self.P))
        self.particles[:,0] = list(bend_angle)

        bend_direction = self.particles[:,1]+np.random.normal(0,self.bend_direction_noise,(self.P))
        self.particles[:,1] = list(bend_direction)
        
        

        if self.print_times == True:
            print("Update processing took: " + str(time.time()-x))

    def resample(self):
        
        #Determine amount of particles to cut
        percent = int(self.P*(self.resample_percent*.01))
        
        #Choose indices, with their weight being the probability being chosen
        all_indices = np.arange(0,self.P,1)
        weights = np.array(self.particles[:,8],dtype='float')
        all_indices = np.random.choice(all_indices,self.P,p=weights)

        #Apply changes
        self.particles = self.particles[all_indices]

    def create_particles(self):
        #Orientations assumed to not be different than standard
        #Structure [[[],[],[] ... []],[[],[],[] ... []],[[],[],[], ... ,[]], ... ,[[],[],[],..., []]] with P lists and N lists inside each P list

        self.particles = []

        for i in range(self.P):
            particle = []

            #Magnet & sensor pose will be generated from bend angle & direction
            segment_bend_angle = .00001
            segment_bend_direction = 270

            magnet_pos = np.array([0.,0.,0.])
            magnet_angle = 0.
            magnet_axis = np.array([0.,0.,0.])

            sensor_pos = np.array([0.,0.,0.])
            sensor_angle = 0.
            sensor_axis = np.array([0.,0.,1.])

            #Weights will be calculated before used
            weight = 0.
            
            #Add data to list, then append to full particles array
            particle.append(segment_bend_angle) #0
            particle.append(segment_bend_direction) #1
            particle.append(magnet_pos) #2
            particle.append(magnet_angle) #3
            particle.append(magnet_axis) #4
            particle.append(sensor_pos) #5
            particle.append(sensor_angle) #6
            particle.append(sensor_axis) #7
            particle.append(weight) #8
            self.particles.append(np.array(particle,dtype='object'))

        self.particles = np.array(self.particles)
        
        return self.particles

if __name__ == "__main__":
    import pandas as pd
    import ast

    df = pd.read_csv("data/processed.csv")

    joints = max(df['joint_index'])+1

    timestep = 0
    timesteps = len(set(df['time']))
    pos = df[['x','y','z']].to_numpy()[joints*timestep:joints*(timestep+1)]
    angle = df['angle'].to_numpy()[joints*timestep:joints*(timestep+1)]
    axis = df[['axis_x','axis_y','axis_z']].to_numpy()[joints*timestep:joints*(timestep+1)]
    
    sensor_data = df[['sensor_x','sensor_y','sensor_z']].to_numpy()
    sensor_data = sensor_data
    sensor_data = np.reshape(sensor_data,(timesteps,joints,3))
    new_sensor_data = []
    pairs = 1
    for i in range(len(sensor_data)):
        new_sensor_data.append(sensor_data[i][::2][0:pairs])
    sensor_pos = pos[::2][0:pairs]
    sensor_angle = angle[::2][0:pairs]
    sensor_axis = axis[::2][0:pairs]
    
    magnet_pos = pos[1::2][0:pairs]
    magnet_angle = angle[1::2][0:pairs]
    magnet_axis = axis[1::2][0:pairs]
    
    loop = True

    x = Filter(pairs,new_sensor_data,timestep)

    x.create_particles()

    if loop==True:
        for i in range(70):
            mag_ipos = df[['x','y','z']].to_numpy()[joints*i:joints*(i+1)][1::2][0:1]
            mag_angle = df['angle'].to_numpy()[joints*i:joints*(i+1)][1::2][0:1]
            mag_axis = df[['axis_x','axis_y','axis_z']].to_numpy()[joints*i:joints*(i+1)][1::2][0:1]
            x.timestep = i
            x.compute_pose()
            x.compute_flux()
            x.reweigh()
            x.predict()
            x.resample()
            x.update()
            print()
            print(mag_ipos)
            print(x.pos)
            print(x.particles[0][0])
            print(x.particles[0][1])
    else:
        x.compute_pose()
        x.compute_flux()
        x.reweigh()
        x.resample()
        x.update()
