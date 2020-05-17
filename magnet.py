import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import numpy as np
import magpylib as magpy
from graph import Graph
from magpylib.source.magnet import Box
from segment2 import Segment, LineSegment, CircleSegment

class Magnet:
    def __init__(self, chain):
        self.chain = chain
        self.magnetization = [0,0,575.4]
        self.dimension = [6.35,6.35,6.35]
        self.magnets = []
        self.sensors = []
        self.sensor_data = []
        self.collection = None

    def SetMagnetParameters(self, magnetization=None, dimension=None):
        if magnetization is not None:
            self.magnetization = magnetization
        if dimension is not None:
            self.dimension = dimension

    def PlaceMagnets(self):

        self.magnets = []
        self.sensors = []

        init_linspace = np.linspace(0,self.chain._segment_count-1,num=self.chain._segment_count)
        init_pose = self.chain.GetPoints(init_linspace)
        init_orient = self.chain.GetOrientations(init_linspace)

        final_linspace = np.linspace(1,self.chain._segment_count,num=self.chain._segment_count)
        final_pose = self.chain.GetPoints(final_linspace)
        final_orient = self.chain.GetOrientations(final_linspace)

        for i in range(self.chain._segment_count):
            #Put sensor at end of line segment
            if(isinstance(self.chain._segments[i],LineSegment)):
                self.sensors.append(magpy.Sensor(pos=init_pose[i],axis=[0,1,0],angle=90))

                rotvec_orient = init_orient[i].as_rotvec()
                if np.array_equal(init_orient[i].apply([1.,0.,0.]),[1.,0.,0.]) is False and np.sum(np.absolute(rotvec_orient)) is not 0:
                    self.sensors[len(self.sensors)-1].rotate(axis=rotvec_orient,angle=(np.linalg.norm(rotvec_orient)*180)/np.pi)

            #Put magnet at end of chain
            if(isinstance(self.chain._segments[i],CircleSegment)):
                self.magnets.append(Box(mag=self.magnetization,dim=self.dimension,pos=final_pose[i],axis=[0,1,0],angle=90))

                rotvec_orient = final_orient[i].as_rotvec()
                if np.array_equal(final_orient[i].apply([1.,0.,0.]),[1.,0.,0.]) is False and np.sum(np.absolute(rotvec_orient)) is not 0:
                    self.magnets[len(self.magnets)-1].rotate(axis=rotvec_orient,angle=(np.linalg.norm(rotvec_orient)*180)/np.pi)

        self.collection = magpy.Collection(self.magnets)
        return self.collection

    def ReadSensors(self):
        self.sensor_data = []
        for i in range(len(self.sensors)):
            self.sensor_data.append(self.sensors[i].getB(self.collection))
        return self.sensor_data


if __name__ == "__main__":
    from chain2 import Chain
    from segment2 import LineSegment,CircleSegment
    from graph import Graph
    import matplotlib.pyplot as plt
    segment_list = []
    for i in range(10):
        segment_list.append(LineSegment(10))
        segment_list.append(CircleSegment(100,np.pi/2,np.pi/2+(.1*i)+.1))

    chain = Chain(segment_list=segment_list)
    t_array = np.linspace(0,len(segment_list),num=200)

    all_points = []
    for i in range(400):
        chain_points = chain.GetPoints(t_array)
        all_points.append(chain_points)

    x = Magnet(chain)
    c = x.PlaceMagnets()

    data = x.ReadSensors()
    print(data)


    plot = Graph(all_points,c,x.sensors)
    plot.SetParameters(show=True,dpi=144,rotate=False)
    plot.Plot()
