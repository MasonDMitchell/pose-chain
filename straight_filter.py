import numpy as np
import math
import magpylib as magpy
import scipy as sci
import time
from chain2 import CompositeSegment
from segment2 import LineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R

class Filter:
    def __init__(self,N,P):

        self.print_times = False

        #Constants
        self.mag = np.array([-575.4,0,0])
        self.dim = np.array([6.35,6.35,6.35])
        self.P = P
        self.MAG = np.array(np.tile(self.mag,(N**2*self.P,1)))
        self.DIM = np.array(np.tile(self.dim,(N**2*self.P,1)))
        self.axis_noise = .1
        self.segment_length = 34.68444
        self.sensor_segment_length = 13.288902
        self.last_best_axis = [0,0,0]
        self.last_velocity = [0,0,0]

        #Inputs
        self.N = N
        self.sensor_data = [0,0,0]

    def compute_pose(self):
       
        #Compute position for first magnet in each particle
        #TODO This might need to go somewhere else in this function
        pos = self.particles[:,5] + self.particles[:,0]*self.segment_length

        #Initialization of initial magnet axis
        init_axis = np.tile([1,0,0],self.P*self.N)
        init_axis = np.reshape(init_axis,(self.P*self.N,3))

        #Creation of array of desired magnet axis
        mag_axis = self.particles[:,0]
        mag_axis = np.concatenate(mag_axis)

        #Computation for rotation required to get desired axis
        cross = np.cross(init_axis,mag_axis)
        cross = cross / np.reshape(np.repeat(np.linalg.norm(cross,axis=1),3),(self.P*self.N,3))
        cross = np.reshape(cross,(self.P,self.N,3))
        
        dot = np.multiply(init_axis,mag_axis).sum(1)
        angle = np.arccos(dot)
        
        angle = np.degrees(angle)
        angle = np.reshape(angle,(self.P,self.N))

        sensor_pos = mag_axis * self.sensor_segment_length + np.concatenate(pos)
        sensor_pos = np.reshape(sensor_pos,(self.P,self.N,3))
        sensor_pos = np.delete(sensor_pos,self.N-1,axis=1)
        sensor_pos = np.append(np.zeros((self.P,1,3)),sensor_pos,axis=1)

        sensor_angle = np.delete(angle,self.N-1,axis=1)
        sensor_angle = np.append(np.zeros((self.P,1)),sensor_angle,axis=1)

        sensor_axis = np.delete(cross,self.N-1,axis=1)
        sensor_axis = np.append(np.zeros((self.P,1,3)),sensor_axis,axis=1)
        
        self.particles[:,2] = list(pos)
        self.particles[:,3] = list(angle)
        self.particles[:,4] = list(cross)
        self.particles[:,5] = list(sensor_pos)
        self.particles[:,6] = list(sensor_angle)
        self.particles[:,7] = list(sensor_axis)

    def compute_flux(self): 

        y = time.time()

        POSm = self.particles[:,2] #Magnet positions
        POSm = np.repeat(POSm,self.N,0)
        POSm = np.concatenate(POSm)
        POSm = np.reshape(POSm,(self.P*self.N*self.N,3))

        ANG = self.particles[:,3] #Magnet angles
        ANG = np.repeat(ANG,(self.N),0)
        ANG = np.concatenate(ANG)
        ANG = ANG.astype("float64")

        AXIS = self.particles[:,4] #Magnet axis
        AXIS = np.repeat(AXIS,self.N,0)
        AXIS = np.concatenate(AXIS)
        AXIS = np.reshape(AXIS,(self.P*self.N*self.N,3))

        POSo = self.particles[:,5] #Sensor positions
        POSo = np.concatenate(POSo)
        POSo = np.repeat(POSo,self.N,0)
        POSo = np.reshape(POSo,(self.P*self.N*self.N,3))
        x = time.time()
        
        self.Bv = magpy.vector.getBv_magnet('box',MAG=self.MAG,DIM=self.DIM,POSo=POSo,POSm=POSm,ANG=[ANG],AX=[AXIS],ANCH=[POSm])

        #print(POSm)
        #print(POSo)
        #print(self.Bv)

        x = time.time()-x
        
        self.Bv = np.reshape(self.Bv,(self.N*self.P,self.N,3))
        self.Bv = np.sum(self.Bv,1)

        ANGo = np.concatenate(self.particles[:,6])
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
        if self.print_times == True:
            print("BV processing took: " + str(x))
            print("Non-BV compute processing took: " + str(y-x))
            print("Compute processing took: " + str(y))
        return self.Bv

    def reweigh(self):
        #TODO Refactor for multiple segments for things in the same chain
        x = time.time() 
        error = np.subtract(self.sensor_data,self.Bv) #3D Difference between sensor and particle data
        error = np.linalg.norm(error,axis=2) #Length of 3D difference
        error = np.sum(error,axis=1)
        error = error*500
        error = -(error*error)
        error = np.exp(error)
        #TODO If null replace with 0
        error = np.divide(error,np.sum(error))
        self.particles[:,8] = error
        if self.print_times == True:
            print("Reweigh processing took: " + str(time.time()-x))

        return error

    def predict(self):
        x = time.time()
        #TODO mean of circular quantities
        maximum = np.argpartition(self.particles[:,8],-1)[-1:]
        
        best = self.particles[maximum]
        self.best_data = self.Bv[maximum][0]
        self.best_pos = best[0][2]
        self.best_angle = best[0][3]
        self.best_axis = best[0][4]

        if self.print_times == True:
            print("Predict processing took: " + str(time.time()-x))

    def update(self):
       
        x = time.time()
        maximum = np.argpartition(self.particles[:,8],-1)[-1:]
        best_axis = self.particles[maximum[0]][0]

        particle_axis = self.particles[:,0]
        particle_axis = np.concatenate(particle_axis)
        particle_axis = np.reshape(particle_axis,(self.P,3))

        velocity = np.subtract(best_axis,self.last_best_axis)
        smoothing = 1
        velocity = ((smoothing*np.array(self.last_velocity))+velocity)/(smoothing+1)

        particle_axis = particle_axis + velocity + np.random.normal(0,self.axis_noise,(self.P,3))

        particle_axis = particle_axis + np.random.normal(0,self.axis_noise,(self.P*self.N,3))
        
        #Normalize to length of 1
        length_axis = np.reshape(np.repeat(np.linalg.norm(particle_axis,axis=1),3),(self.P*self.N,3))
        particle_axis = particle_axis/length_axis
        particle_axis = np.reshape(particle_axis,(self.P,self.N,3))
        self.particles[:,0] = list(particle_axis)

        self.last_best_axis = best_axis
        self.last_velocity = velocity
        if self.print_times == True:
            print("Update processing took: " + str(time.time()-x))

    def resample(self):
        
        #Determine amount of particles to cut
        cut_particles = self.P//2
        
        #Choose indices, with their weight being the probability being chosen
        all_indices = np.arange(0,self.P,1)
        weights = np.array(self.particles[:,8],dtype='float')

        maximum = np.argpartition(self.particles[:,8],-cut_particles)[-cut_particles:]
        minimum = np.argpartition(self.particles[:,8],cut_particles)[:cut_particles]
        self.particles[minimum] = self.particles[maximum]

        #Apply changes
        self.particles = self.particles[all_indices]

    def create_particles(self,sensor_init_pos,sensor_init_angle,sensor_init_axis,magnet_init_pos,magnet_init_angle,magnet_init_axis):
        #Orientations assumed to not be different than standard
        #Structure [[[],[],[] ... []],[[],[],[] ... []],[[],[],[], ... ,[]], ... ,[[],[],[],..., []]] with P lists and N lists inside each P list

        self.particles = []

        for i in range(self.P):
            particle = []

            #Magnet & sensor pose will be generated from bend angle & direction
            segment_axis = np.tile([1,0.,0.00001],(self.N,1))
            segment_bend_direction = 270
            
            magnet_pos = np.array(magnet_init_pos)
            magnet_angle = magnet_init_angle
            magnet_axis = np.reshape(np.tile([0,0,1],self.N),(self.N,3))
            
            sensor_pos = sensor_init_pos
            sensor_angle = sensor_init_angle
            sensor_axis = np.reshape(np.tile([0,0,1],self.N),(self.N,3))

            #Weights will be calculated before used
            weight = 0
            
            #Add data to list, then append to full particles array
            particle.append(segment_axis) #0
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
    from matplotlib import pyplot as plt
    import pickle
    import ast

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

    x = Filter(pairs,new_sensor_data,sensor_pos,sensor_angle,sensor_axis,magnet_pos,magnet_angle,magnet_axis)

    x.create_particles()

    error = []
    pos_diff = []
    truth_pos_diff = []

    last_pos = [86.711098,0,0]
    truth_last_pos = [86.711098,0,0]

    x.segment_length = 86.711098

    magnet_data = []

    if loop==True:
        testing = time.time()
        for i in range(timesteps):
            #print(i)
            mag_ipos = df[['x','y','z']].to_numpy()[joints*i:joints*(i+1)][1::2][0:pairs]
            mag_angle = df['angle'].to_numpy()[joints*i:joints*(i+1)][1::2][0:pairs]
            mag_axis = df[['axis_x','axis_y','axis_z']].to_numpy()[joints*i:joints*(i+1)][1::2][0:pairs]
            x.timestep = i
            x.test = mag_ipos[0]
            x.test1= mag_angle[0]
            if(np.array_equal(mag_axis[0],np.array([0.,0.,0.]))):
                    x.test2 = np.array([1,0,0])
            else:
                x.test2 = mag_axis[0]

            #x.segment_length = np.linalg.norm(mag_ipos)
            x.compute_pose()
            x.compute_flux()
            x.reweigh()
            x.predict()
            maximum = np.argpartition(x.particles[:,8],-1)[-1:]
            minimum = np.argpartition(x.particles[:,8],1)[:1]
            
                    
            print("----------------------")
            print("Maximum weight" + str(x.particles[maximum[0]][8]))
            print("Minimum weight" + str(x.particles[minimum[0]][8]))
            print()
            print("Minimum weight sensor reading:" + str(x.Bv[minimum[0]]))
            print("Maximum weight sensor reading:" + str(x.Bv[maximum[0]]))
            print("Real sensor reading: " + str(x.sensor_data[x.timestep]))
            print()
            print("Minimum weight magnet pos:" + str(x.particles[minimum[0]][2]))
            print("Maximum weight magnet pos:" + str(x.particles[maximum[0]][2]))
            print("Summed pos:" + str(x.pos))
            print("Real pos length: " + str(np.linalg.norm(mag_ipos)))
            print("Real magnet pos: " + str(mag_ipos))
            print()
            

            error.append(np.linalg.norm(np.subtract(x.particles[maximum[0]][2],mag_ipos)))
            #pos_diff.append([x.particles[maximum[0]][2][0]-last_pos[0],x.particles[maximum[0]][2][1]-last_pos[1],x.particles[maximum[0]][2][2]-last_pos[2]])
            #truth_pos_diff.append([mag_ipos[0][0]-truth_last_pos[0],mag_ipos[0][1]-truth_last_pos[1],mag_ipos[0][2]-truth_last_pos[2]])
            #truth_last_pos = mag_ipos[0]
            #last_pos = x.particles[maximum[0]][2]
            
            '''
            print(x.particles[maximum[0]][2])
            print(x.particles[maximum[0]][3])
            print(x.particles[maximum[0]][4])
            print("-----------------")
            print(x.particles[minimum[0]][2])
            print(x.particles[minimum[0]][3])
            print(x.particles[minimum[0]][4])
            '''
            #magnet_data.append([x.particles[minimum[0]][2],x.particles[minimum[0]][3],x.particles[minimum[0]][4]])
            #magnet_data.append([x.particles[maximum[0]][2],x.particles[maximum[0]][3],x.particles[maximum[0]][4]])
            magnet_data.append([mag_ipos[0],mag_angle,mag_axis[0]])

            x.resample()
            x.update()

    else:
        x.compute_pose()
        x.compute_flux()
        x.reweigh()
        x.resample()
        x.update()

    print(str(380/(time.time()-testing)) + "Hz")

    pickle.dump(magnet_data, open( "data/test.p", "wb" ) )

    '''
    pos_diff = np.array(pos_diff)
    truth_pos_diff = np.array(truth_pos_diff)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(pos_diff[:,0]-truth_pos_diff[:,0])
    axs[0, 0].set_title('X vel')
    axs[0, 1].plot(pos_diff[:,1]-truth_pos_diff[:,1], 'tab:orange')
    axs[0, 1].set_title('Y vel')
    axs[1, 0].plot(pos_diff[:,2]-truth_pos_diff[:,2], 'tab:green')
    axs[1, 0].set_title('Z vel')
    axs[1, 1].plot(np.sum(pos_diff,1)-np.sum(truth_pos_diff,1), 'tab:red')
    axs[1, 1].set_title('Total_Vel')
    #plt.show()
    '''
    plt.plot(error)
    plt.title("Error vs Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Error(mm)")
    plt.show()
    
