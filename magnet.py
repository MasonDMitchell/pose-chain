import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import numpy as np
import magpylib as magpy
from animation import Animate
from magpylib.source.magnet import Box

class Magnet:
    def __init__(self, chain):
        self.chain = chain
        self.magnetization = [0,0,575.4]
        self.dimension = [6.35,6.35,6.35]
        self.magnets = []
        self.sensors = []

    def SetMagnetParameters(self, magnetization=None, dimension=None):
        if magnetization is not None:
            self.magnetization = magnetization
        if dimension is not None:
            self.dimension = dimension

    def PlaceMagnets(self):
        #final_orientation returns final rotation vector of the segment
        #Get_Points([0]) for beginning of segment
        #Get_Points([1]) for end of segment
        #GetOrientations([0]) gives scipy rotation for beginning of segment

        self.magnets = []
        self.sensors = []

        init_linspace = np.linspace(0,self.chain._segment_count-1,num=self.chain._segment_count) 
        init_pose = self.chain.GetPoints(init_linspace)
        init_orient = self.chain.GetOrientations(init_linspace)

        final_linspace = np.linspace(1,self.chain._segment_count,num=self.chain._segment_count)
        final_pose = self.chain.GetPoints(final_linspace)
        final_orient = self.chain.GetOrientations(final_linspace)

        for i in range(self.chain._segment_count):
            self.sensors.append(magpy.Sensor(pos=init_pose[i]))
            self.magnets.append(Box(mag=self.magnetization,dim=self.dimension,pos=final_pose[i]))
            self.sensors[i].setOrientation(axis=init_orient[i].apply([1,0,0]),angle=(len(init_orient[i])*180)/math.pi)
            self.magnets[i].setOrientation(axis=final_orient[i].apply([1,0,0]),angle=(len(final_orient[i])*180)/math.pi)

        return magpy.Collection(self.magnets)

if __name__ == "__main__":
    from chain2 import Chain
    from segment2 import LineSegment,CircleSegment

    segment_list = []
    segment_list.append(LineSegment(80))
    segment_list.append(CircleSegment(100,np.pi/4,-np.pi/2))

    chain = Chain(segment_list=segment_list)
    t_array = np.linspace(0,2,num=20)

    chain_points = chain.GetPoints(t_array)

    x = Magnet(chain)
    c = x.PlaceMagnets()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    magpy.displaySystem(c,subplotAx=ax,suppress=True,sensors=x.sensors)

    ax.plot(chain_points[:,0],
            chain_points[:,1],
            chain_points[:,2])
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_zlim(-100,100)

    plt.show()
