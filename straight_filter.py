import numpy as np
import math
import magpylib as magpy
import scipy as sci
import time
from scipy.spatial.transform import Rotation as R

class Filter:
    def __init__(self,chain,noise):

        #Constants
        #sim
        self.mag = np.array([-575.4,0,0])
        #real
        #self.mag = np.array([-151,0,0])
        #self.mag = np.array([-60,-30,30])

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
        self.magnet_array = np.arange(1,self.N+1,1)
        self.sensor_array = np.arange(.5,self.N+.5,1)

    def angle_axis2(self,orientation): 
        rotvecs = []
        for i in range(len(orientation)):
            #Generate rotvecs and angles for rotation
            rotvecs.append(orientation[i].as_rotvec())

        rotvecs = np.array(rotvecs)

        angles = np.linalg.norm(rotvecs,axis=2)
        
        #Get indices that need no rotation
        zero = np.where(angles == 0.0)[0]
        #Replace rotvecs that need no rotation
        rotvecs[zero] = [1,0,0]
        
        #Ensure rotvecs have length 1
        rotvecs = np.divide(rotvecs,np.reshape(np.repeat(np.linalg.norm(rotvecs,axis=2),3),(self.N,self.P,3)))
        
        #Change radians to degrees for processing
        angles = np.degrees(angles)

        return angles, rotvecs
        
    def compute_flux(self): 

        POSm = self.chain.GetPoints(self.magnet_array)
        POSm = POSm.flatten()
        POSm = np.tile(POSm,self.N)
        POSm = np.reshape(POSm,(self.P*self.N*self.N,3))

        SPINm = self.chain.GetOrientations(self.magnet_array)
        ANG, AXIS = self.angle_axis2(SPINm)
        
        ANG = np.reshape(ANG,(self.P,self.N))
        ANG = np.repeat(ANG,(self.N),0)
        ANG = np.concatenate(ANG)
        
        AXIS = AXIS.flatten()
        AXIS = np.tile(AXIS,self.N)
        AXIS = np.reshape(AXIS,(self.P*self.N*self.N,3))

        POSo = self.chain.GetPoints(self.sensor_array) #Sensor positions
        POSo = np.concatenate(POSo)
        POSo = np.repeat(POSo,self.N,0)
        POSo = np.reshape(POSo,(self.P*self.N*self.N,3))
        x = time.time()
        
        self.Bv = magpy.vector.getBv_magnet('box',MAG=self.MAG,DIM=self.DIM,POSo=POSo,POSm=POSm,ANG=[ANG],AX=[AXIS],ANCH=[POSm])

        # FIRST # OF SENSORS
        # MIDDLE # OF MAGNETS
        #LAST # OF PARTICLE
        #print("----------------------")
        #print(self.Bv)
        self.Bv = np.reshape(self.Bv,(self.N,self.N,self.P,3))
        #print(self.Bv)
        self.Bv = np.sum(self.Bv,1)
        #print(self.Bv)
        self.Bv = np.reshape(self.Bv,(self.N*self.P,3))
        #print(self.Bv)

        SPINo = self.chain.GetOrientations(self.sensor_array)
        ANGo, AXISo = self.angle_axis2(SPINo)
        
        ANGo = np.radians(ANGo)
        ANGo = np.repeat(ANGo,3,0)
        ANGo = ANGo.flatten()

        AXISo = np.concatenate(AXISo)
        AXISo = AXISo.flatten()

        rotvecs = np.multiply(ANGo,AXISo)
        rotvecs = np.reshape(rotvecs,(self.N*self.P,3))

        r = R.from_rotvec(rotvecs)
 
        self.Bv = r.apply(self.Bv,inverse=True)
        self.Bv = np.reshape(self.Bv,(self.N,self.P,3))
        self.Bv = np.swapaxes(self.Bv,0,1)
         
        #print("Sensor Data: {}".format(np.squeeze(self.sensor_data)))
        #print("Bv Data: {}".format(np.squeeze(self.Bv)))
        return self.Bv

    def reweigh(self):

        error = np.subtract(self.sensor_data,self.Bv) #3D Difference between sensor and particle data
        error = np.linalg.norm(error,axis=2) #Length of 3D difference
        
        #print("ERROR VALUE:",error)
        error = np.sum(error,axis=1)
        error = error*10
        error = -(error*error)
        #error = np.exp(error) #<-- THIS IS THE THING RETURNING 0
        #print("ERROR:",error)

        logsum = sci.special.logsumexp(error)

        #print("LOGSUM:",logsum)
        

        #print("ERROR-LOGSUM:",(error-logsum))
        #print("EXP OF ERROR-LOGSUM:",np.exp(error-logsum))
        #print("SUM OF EXP OF ERROR-LOGSUM:",np.sum(np.exp(error-logsum)))
        #print("ERROR - ERROR SUM (Prevous Algorithm):",np.divide(error,np.sum(error)))    

        error = np.exp(error-logsum)

        #error = np.divide(error,np.sum(error))
        self.weights = error
        #print("Weights:",self.weights)

        return error

    def predict(self):
        #TODO mean of circular quantities for bend direction
        #Predict from best particle, pose is probably the only useful one
        points = self.chain.GetPoints(self.magnet_array)[0]
        bend_angle, bend_direction = self.chain.GetParameters()
        #bend_direction = np.unwrap(bend_direction)
        index = np.argmax(self.weights)

        percentage = self.P//10
        largest_val = np.argpartition(self.weights,-percentage)
        
        total_weights_bounded = np.divide(self.weights[largest_val[-percentage:]],np.sum(self.weights[largest_val[-percentage:]]))

        total_weights_pos_by_weight = points[largest_val[-percentage:]] * np.reshape(np.repeat(total_weights_bounded,3),(percentage,3))
        total_weights_pos_sum = np.sum(total_weights_pos_by_weight,axis=0)

        bend_angle = np.average(bend_angle[largest_val[-percentage:]])
        bend_direction = np.arctan2(np.sum(np.sin(bend_direction)),np.sum(np.cos(bend_direction)))

        return total_weights_pos_sum,bend_angle,bend_direction

    def closest_point(self,pos):
        points = self.chain.GetPoints(self.magnet_array)[0]
        error = np.subtract(points,pos)
        error = np.linalg.norm(error,axis=1)
        index = np.argmin(error)

        print(points[index])
        return

    def update(self):

        self.params = self.noise(self.chain.GetParameters(),.01)
        self.chain.SetParameters(*self.params)

    def update2(self):
        self.params = self.chain.GetParameters()
        noise_alpha = np.random.normal(0,.1,self.P)
        new_param_alpha = self.params[0] + noise_alpha

        noise_beta = np.random.normal(0,.1,self.P)
        new_param_beta = self.params[1] + noise_beta

        self.params = np.array([new_param_alpha,new_param_beta])
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

        ''' 
        percentage = self.P//2
        smallest_val = np.argpartition(self.weights,percentage)

        largest_val = np.argpartition(self.weights,-percentage)
        self.params[0][smallest_val[:percentage]] = self.params[0][largest_val[-percentage:]] 
        self.params[1][smallest_val[:percentage]] = self.params[1][largest_val[-percentage:]] 
        self.weights[smallest_val[:percentage]] = self.weights[largest_val[-percentage:]] 
        
        '''

if __name__ == "__main__":
    import pandas as pd
    from matplotlib import pyplot as plt
    from tools import createChain,spiral_test,noise,noise_rejection
    import sys, os

   # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__
    
    #Spiral for testing
    blockPrint()
    flux, sensor_pos, magnet_pos, alpha, beta = spiral_test(segments=1,
                                                    bend_angle=0,
                                                    bend_direction=0,
                                                    bend_length=14,
                                                    straight_length=0)

    print()
    print("END OF SPIRAL")
    print()
    enablePrint()
    chain = createChain(particles=1000,
                        segments=1,
                        bend_angle=0,
                        bend_direction=0,
                        bend_length=14,
                        straight_length=0)
    x = Filter(chain,noise)

    difference = []
    '''
    ax = []
    ay = []
    bx = []
    by = []
    '''

    test = time.time()
    for i in range(len(flux)):
        x.sensor_data = flux[i]
        x.compute_flux()
        x.reweigh()
        x.resample()
        #print(flux[i])
        #print(x.chain.GetPoints(x.magnet_array)[:5])
        #print("Predict:",x.predict()[0])
        #print("Actual:",magnet_pos[i])
   
        #difference.append(np.linalg.norm(np.subtract(x.predict()[0],magnet_pos[i])))
        #difference.append(np.subtract(x.predict()[1],alpha[i]))
        '''
        ax.append(x.predict()[1]/1.57 * np.cos(x.predict()[2]))
   ain = createChain(particles=1000,
                        segments=1,
                        bend_angle=0,
                        bend_direction=0,
                        bend_length=14,
                        straight_length=0)
    ay.append(x.predict()[1]/1.57 * np.sin(x.predict()[2]))
        bx.append(alpha[i]/1.57*np.cos(beta[i]))
        by.append(alpha[i]/1.57*np.sin(beta[i]))
        '''

        x.update()
    
    print(len(flux)/(time.time()-test))
    '''
    plt.style.use("ggplot")
    plt.title("Parameter Space Plot Actual & Predicted")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.plot(bx,by,linewidth=4,color='black')
    plt.plot(ax,ay,color='yellow')
    plt.show()

    difference = [np.subtract(ax,bx),np.subtract(ay,by)]
    difference = np.linalg.norm(difference,axis=0)

    plt.title("Parameter Space Error")
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.plot(difference,label="Error")
    plt.plot(np.repeat(np.mean(difference),len(flux)),label="Mean: " + str(np.round(np.mean(difference),4)))
    plt.legend()
    plt.show()

    '''


"""
    plt.style.use("ggplot")
    plt.title("Bend Angle Filter Error")
    plt.ylabel("Predicted - Actual (rad)")
    plt.xlabel("Filter Iteration")
    plt.plot(difference,label="Error (mm)")
    plt.plot(np.repeat(np.std(difference),len(flux)),color='blue')
    plt.plot(np.repeat(-np.std(difference),len(flux)),color='blue',label="Standard Deviation: " + str(np.round(np.std(difference),4)))
    plt.legend()
    plt.show()
"""


'''

    plt.style.use("ggplot")
    plt.title("3D Position Filter Error")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Filter Iteration")
    plt.plot(difference,label="Error (mm)")
    plt.plot(np.repeat(np.mean(difference),len(flux)),label="Mean: " + str(np.round(np.mean(difference),4))+"mm")
    plt.legend()
    plt.show()
'''
