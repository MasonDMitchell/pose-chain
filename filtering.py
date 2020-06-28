import numpy as np
import magpylib as magpy
import scipy as sci
import time
from scipy.spatial.transform import Rotation as R

#Particle dimension 5N, needs N magnets, N sensors (need to know position of sensors

#Input dimension 3N: xyz 
class Filter:
    def __init__(self,N,sensor_init_pos,sensor_init_angle,sensor_init_axis,magnet_init_pos,magnet_init_angle,magnet_init_axis,sensor_data,timestep):

        self.timestep = timestep

        #Constants
        self.mag = np.array([0,0,-575.4])
        self.dim = np.array([6.35,6.35,6.35])
        self.P = 10000
        self.loop_MAG = np.tile(self.mag,(N,1)) #Deprecated
        self.loop_DIM = np.tile(self.dim,(N,1)) #Deprecated
        self.MAG = np.array(np.tile(self.mag,(N**2*self.P,1)))
        self.DIM = np.array(np.tile(self.dim,(N**2*self.P,1)))
        self.init_weight = .25
        self.init_pos_noise = 0
        self.init_angle_noise = 0
        self.init_axis_noise = 0
        self.pos_noise = 3
        self.angle_noise = 5
        self.axis_noise = 2
        self.sampling_ratio = .4 #This is so we don't have to sample all particles
        self.segment_length = 5.315559

        #Inputs
        self.N = N
        self.sensor_init_pos = np.array(sensor_init_pos)
        self.sensor_init_angle = np.array(sensor_init_angle)
        self.sensor_init_axis = np.array(sensor_init_axis)
        self.magnet_init_pos = np.array(magnet_init_pos)
        self.magnet_init_angle = np.array(magnet_init_angle)
        self.magnet_init_axis = np.array(magnet_init_axis)
        self.sensor_data = np.array(sensor_data)

    def compute(self): 

        y = time.time()

        POSo = self.particles[:,0] #Sensor positions
        POSo = np.concatenate(POSo)
        POSo = np.repeat(POSo,self.N,0)

        POSm = self.particles[:,3] #Magnet positions
        POSm = np.repeat(POSm,self.N,0)
        POSm = np.concatenate(POSm)
        
        ANG = self.particles[:,4] #Magnet angles
        ANG = np.repeat(ANG,self.N,0)
        ANG = np.concatenate(ANG)
        ANG = ANG.flatten()

        AXIS = self.particles[:,5] #Magnet axis
        AXIS = np.repeat(AXIS,self.N,0)
        AXIS = np.concatenate(AXIS) 

        x = time.time()
        
        self.Bv = magpy.vector.getBv_magnet('box',MAG=self.MAG,DIM=self.DIM,POSo=POSo,POSm=POSm,ANG=[ANG],AX=[AXIS],ANCH=[POSm])

        x = time.time()-x
    
        self.Bv = np.reshape(self.Bv,(self.N*self.P,self.N,3))
        self.Bv = np.sum(self.Bv,1)

        ANGo = np.concatenate(self.particles[:,1])
        ANGo = np.radians(ANGo)
        ANGo = np.repeat(ANGo,3,0)
        ANGo = ANGo.flatten()

        AXISo = np.concatenate(self.particles[:,2])
        AXISo = AXISo.flatten()

        rotvecs = np.multiply(ANGo,AXISo)
        rotvecs = np.reshape(rotvecs,(self.N*self.P,3))

        r = R.from_rotvec(rotvecs)
        self.Bv = r.apply(self.Bv,inverse=True)
        self.Bv = np.reshape(self.Bv,(self.P,self.N,3))

        y = time.time()-y

        print("BV processing took: " + str(x))
        print("Non-BV compute processing took: " + str(y-x))
        print("Compute processing took: " + str(y))

        return self.Bv

    def reweigh(self):
        x = time.time() 

        error = np.subtract(self.sensor_data[self.timestep],self.Bv) #3D Difference between sensor and particle data
        error = np.linalg.norm(error,axis=2) #Length of 3D difference
        error = -(error*error)
        error = np.exp(error)
        error = np.sum(error,axis=1)
        error = np.reshape(error,(self.P))
        
        error = np.divide(error,np.sum(error))
        self.particles[:,6] = error
        print("Reweigh processing took: " + str(time.time()-x))

        return error

    def predict(self):
        x = time.time()

        self.pos = np.sum(np.multiply(self.particles[:,6],self.particles[:,3]))
        self.angle = np.sum(np.multiply(self.particles[:,6],self.particles[:,4]))
        self.axis = np.sum(np.multiply(self.particles[:,6],self.particles[:,5]))

        print("Predict processing took: " + str(time.time()-x))

    def isclose(self,position):
        error = np.reshape(np.concatenate(self.particles[:,3]),(self.P,self.N,3))
        error = np.subtract(position,error)
        error = np.linalg.norm(error,axis=2)
        error = np.concatenate(error)
        minimum = np.argpartition(error,1)[:1]
        #print(minimum)
        #print(error[minimum])
        #print(self.particles[minimum][0][3])
        #print(position)

    def update(self):
        
        x = time.time()

        magnet_pos_noise = np.reshape(np.concatenate(self.particles[:,3]),(self.P,self.N,3))#Magnet pos shaped
        magnet_pos_noise = magnet_pos_noise + np.random.normal(0,self.pos_noise,(self.P,self.N,3))#Magnet pos w/ noise
        self.particles[:,3] = list(magnet_pos_noise) #Magnet pos updated

        magnet_angle_noise = np.reshape(np.concatenate(self.particles[:,4]),(self.P,self.N,1))#Magnet angle shaped
        magnet_angle_noise = magnet_angle_noise + np.random.normal(0,self.angle_noise,(self.P,self.N,1))#Magnet angle w/ noise
        magnet_angle_noise = np.mod(magnet_angle_noise,360)
        self.particles[:,4] = list(magnet_angle_noise)

        magnet_axis_noise = np.reshape(np.concatenate(self.particles[:,5]),(self.P,self.N,3))#Magnet axis shaped
        magnet_axis_noise = np.reshape(magnet_axis_noise.flatten() / np.repeat(np.linalg.norm(magnet_axis_noise,axis=2).flatten(),3),(self.P,self.N,3))#Magnet axis normalized
        magnet_axis_noise = magnet_axis_noise + np.random.normal(0,self.axis_noise,(self.P,self.N,3))#Magnet axis w/ noise
        magnet_axis_noise = np.reshape(magnet_axis_noise.flatten() / np.repeat(np.linalg.norm(magnet_axis_noise,axis=2).flatten(),3),(self.P,self.N,3))#Magnet axis normalized
        self.particles[:,5] = list(magnet_axis_noise)
        
        sensor_axis = np.delete(magnet_axis_noise,self.N-1,axis=1)
        sensor_axis = np.insert(sensor_axis,0,[1,0,0],axis=1)
        self.particles[:,2] = list(sensor_axis)

        sensor_pos = np.reshape(np.concatenate(self.particles[:,3]),(self.P,self.N,3))
        sensor_pos = sensor_pos + (sensor_axis*self.segment_length)
        sensor_pos = np.delete(sensor_pos,self.N-1,axis=1)
        sensor_pos = np.insert(sensor_pos,0,[0,0,0],axis=1)
        self.particles[:,0] = list(sensor_pos)
        
        sensor_angle = np.delete(magnet_angle_noise,self.N-1,axis=1)
        sensor_angle = np.insert(sensor_angle,0,0,axis=1)
        self.particles[:,1] = list(sensor_angle)

        print("Update processing took: " + str(time.time()-x))

    def resample(self):
        percent = 10
        percent = int(self.P*(percent*.01))
        all_indices = np.arange(0,self.P,1)
        weights = np.array(self.particles[:,6],dtype='float')
        all_indices = np.random.choice(all_indices,self.P,p=weights)
        self.particles = self.particles[all_indices]

        
    def loop_compute(self):
        x = time.time()
        for j in range(self.P):
            for i in range(self.N):
                POSo = np.tile(self.particles[:,0][j][i],(self.N,1))
                POSm = self.particles[:,3][j]
                ANG = self.particles[:,4][j]
                AXIS = self.particles[:,5][j]
                Bv = magpy.vector.getBv_magnet('box',self.loop_MAG,self.loop_DIM,POSo,POSm, ANG, AXIS,POSm)
        print(time.time()-x)

    def create_particles(self):
        #Orientations assumed to not be different than standard
        #Structure [[[],[],[] ... []],[[],[],[] ... []],[[],[],[], ... ,[]], ... ,[[],[],[],..., []]] with P lists and N lists inside each P list

        self.particles = []

        for i in range(self.P):
            particle = []

            gaussian_sensor_pos = self.sensor_init_pos + np.random.normal(0,self.init_pos_noise,(self.N,3))
            gaussian_sensor_angle = self.sensor_init_angle
            #gaussian_sensor_axis = self.sensor_init_axis + np.random.normal(0,self.init_axis_noise,(self.N,3))
            gaussian_sensor_axis = np.array([[0,1,0]]*self.N)

            gaussian_magnet_pos = self.magnet_init_pos + np.random.normal(0,self.init_pos_noise,(self.N,3))
            #TODO fix gaussian angles
            gaussian_magnet_angle = self.magnet_init_angle
            gaussian_magnet_axis = np.array([[0,1,0]]*self.N)

            sensor = [gaussian_sensor_pos,gaussian_sensor_angle,gaussian_sensor_axis] #pos, angle, axis
            magnet = [gaussian_magnet_pos,gaussian_magnet_angle,gaussian_magnet_axis] #pos, angle, axis
            weight = self.init_weight
             
            particle.append(gaussian_sensor_pos) 
            particle.append(gaussian_sensor_angle) 
            particle.append(gaussian_sensor_axis) 
            particle.append(gaussian_magnet_pos) 
            particle.append(gaussian_magnet_angle) 
            particle.append(gaussian_magnet_axis)
            particle.append(weight)
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
    #print(new_sensor_data[timestep:(timestep+1)])
    sensor_pos = pos[::2][0:pairs]
    sensor_angle = angle[::2][0:pairs]
    sensor_axis = axis[::2][0:pairs]
    
    magnet_pos = pos[1::2][0:pairs]
    magnet_angle = angle[1::2][0:pairs]
    magnet_axis = axis[1::2][0:pairs]
    
    loop = True

    x = Filter(pairs,sensor_pos,sensor_angle,sensor_axis,magnet_pos,magnet_angle,magnet_axis,new_sensor_data,timestep)
    x.create_particles()
    if loop==True:
        for i in range(50):
            mag_ipos = df[['x','y','z']].to_numpy()[joints*i:joints*(i+1)][1::2][0:1]
            mag_angle = df['angle'].to_numpy()[joints*i:joints*(i+1)][1::2][0:1]
            x.timestep = i
            x.compute()
            x.reweigh()
            x.predict()
            x.isclose(mag_ipos)
            x.resample()
            x.update()
            #print()
            #print(x.particles[:,3])
            print(x.pos)
            print(mag_ipos)
            #test = np.subtract(mag_ipos,x.particles[:,3])
            #print(x.angle)
    else:
        x.compute()
        x.reweigh()
        x.predict()
        x.resample()
        x.update()
