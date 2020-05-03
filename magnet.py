import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class Magnet:
    def __init__(self, chain):
        self.chain = chain
        self.magnetization = [0,0,575.4]
        self.dimension = [6.35,6.35,6.35]

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

        init_linspace = np.linspace(0,self.chain._segment_count-1,num=self.chain._segment_count) 
        init_pose = self.chain.GetPoints(init_linspace)
        init_orient = self.chain.GetOrientations(init_linspace)

        final_linspace = np.linspace(1,self.chain._segment_count,num=self.chain._segment_count)
        final_pose = self.chain.GetPoints(final_linspace)
        final_orient = self.chain.GetOrientations(final_linspace)

if __name__ == "__main__":
    from chain2 import Chain
    from segment2 import LineSegment,CircleSegment

    segment_list = []
    segment_list.append(LineSegment(10))
    segment_list.append(CircleSegment(20,np.pi/4,-np.pi/2))

    chain = Chain(segment_list=segment_list)

    x = Magnet(chain)
    x.PlaceMagnets()
