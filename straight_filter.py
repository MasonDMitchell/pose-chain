import numpy as np
import math
import magpylib as magpy
import scipy as sci
import time
from scipy.spatial.transform import Rotation as R

class Filter:
    def __init__(self,chain,noise):

        #Constants
        self.mag = np.array([-575.4,0,0])
        self.dim = np.array([6.35,6.35,6.35])
        self.N = chain.segment_count
        self.P = len(chain.GetParameters()[0])
        self.MAG = np.array(np.tile(self.mag,(self.N**2*self.P,1)))
        self.DIM = np.array(np.tile(self.dim,(self.N**2*self.P,1)))
        self.chain = chain
        self.noise = noise
        self.sensor_data = [0,0,0]
        self.weights = []
        self.params = self.chain.GetParameters()

    def angle_axis(self,orientation):
        rotvec = orientation.as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle == 0.0:
            return np.array([[0.0],[1,0,0]],dtype='object')
        axis = np.divide(rotvec,np.repeat(np.linalg.norm(rotvec),3))
        angle = np.degrees(angle)

        return np.array([[angle],axis],dtype='object')

    def compute_flux(self): 

        magnet_array = np.arange(1,self.N+1,1)
        sensor_array = np.arange(.5,self.N+.5,1)

        POSm = self.chain.GetPoints(magnet_array)[:,0]
        POSm = np.repeat(POSm,self.N,0)
        #POSm = np.concatenate(POSm)
        #POSm = np.reshape(POSm,(self.P*self.N*self.N,3))

        SPINm = self.chain.GetOrientations(magnet_array)
        SPINm = np.reshape(SPINm,self.P)
        SPINm = list(map(self.angle_axis,SPINm))
        SPINm = np.array(SPINm)

        ANG = SPINm[:,0] #Magnet angles
        ANG = np.repeat(ANG,(self.N),0)
        ANG = np.concatenate(ANG)
        ANG = ANG.astype("float64")

        AXIS = SPINm[:,1] #Magnet axis
        AXIS = np.repeat(AXIS,self.N,0)
        AXIS = np.concatenate(AXIS)
        AXIS = np.reshape(AXIS,(self.P*self.N*self.N,3))

        POSo = self.chain.GetPoints(sensor_array)[:,0] #Sensor positions
        POSo = np.concatenate(POSo)
        POSo = np.repeat(POSo,self.N,0)
        POSo = np.reshape(POSo,(self.P*self.N*self.N,3))
        x = time.time()
        
        self.Bv = magpy.vector.getBv_magnet('box',MAG=self.MAG,DIM=self.DIM,POSo=POSo,POSm=POSm,ANG=[ANG],AX=[AXIS],ANCH=[POSm])
 
        self.Bv = np.reshape(self.Bv,(self.N*self.P,self.N,3))
        self.Bv = np.sum(self.Bv,1)

        SPINo = self.chain.GetOrientations(sensor_array)
        SPINo = np.reshape(SPINo,self.P)
        SPINo = np.array(list(map(self.angle_axis,SPINo)))

        ANGo = np.concatenate(SPINo[:,0])
        ANGo = np.radians(ANGo)
        ANGo = np.repeat(ANGo,3,0)
        ANGo = ANGo.flatten()

        AXISo = SPINo[:,1]
        AXISo = np.concatenate(AXISo)
        AXISo = AXISo.flatten()

        rotvecs = np.multiply(ANGo,AXISo)
        rotvecs = np.reshape(rotvecs,(self.N*self.P,3))

        r = R.from_rotvec(rotvecs)
        
        self.Bv = r.apply(self.Bv,inverse=True)
        self.Bv = np.reshape(self.Bv,(self.P,self.N,3))

        return self.Bv

    def reweigh(self):
        #TODO Refactor for multiple segments for things in the same chain
        error = np.subtract(self.sensor_data,self.Bv) #3D Difference between sensor and particle data
        error = np.linalg.norm(error,axis=2) #Length of 3D difference
        error = np.sum(error,axis=1)
        error = error*100
        error = -(error*error)
        error = np.exp(error)
        #TODO If null replace with 0
        error = np.divide(error,np.sum(error))
        self.weights = error

        return error

    def predict(self):
        #TODO mean of circular quantities for bend angle & direction
        #Predict from best particle, pose is probably the only useful one
        x=1
    def update(self):

        self.params = self.noise(self.chain.GetParameters(),.1)
        self.chain.SetParameters(*self.params)

    def resample(self):

        #Choose particles to keep based on probability bag
        index_array = np.arange(0,self.P,1)
        index_array = np.random.choice(
                a=index_array,
                size=self.P,
                p=self.weights)

        self.params = self.params[:,index_array]

        self.chain.SetParameters(*self.params)


if __name__ == "__main__":
    import pandas as pd
    from matplotlib import pyplot as plt
    from tools import createChain,noise

    df = pd.read_csv("data/processed.csv")

    joints = max(df['joint_index'])+1

    timestep = 0
    timesteps = len(set(df['time']))
    pos = df[['x','y','z']].to_numpy()[joints*timestep:joints*(timestep+1)]
    angle = df['angle'].to_numpy()[joints*timestep:joints*(timestep+1)]
    axis = df[['axis_x','axis_y','axis_z']].to_numpy()[joints*timestep:joints*(timestep+1)]
    
    sensor_data = df[['sensor_x','sensor_y','sensor_z']].to_numpy()
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

    chain = createChain(10,1,0,0,86.710381,10)

    x = Filter(chain,noise)

    if loop==True:
        test = time.time()
        for i in range(timesteps):
            x.sensor_data = new_sensor_data[i]
            x.compute_flux()
            x.reweigh()
            #x.predict()
            x.resample()
            x.update()
        print(timesteps/(time.time()-test))
    else:
        x.compute_flux()
        x.reweigh()
        x.resample()
        x.update()

    
