
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class Segment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractproperty
    def final_location(self):
        pass

    # returns the final rotation vector of the segment
    @abstractproperty
    def final_orientation(self):
        pass

    # returns x,y,z coordinates for each t in t_array
    # t varies from 0 to 1
    # t_array can contain a single element
    @abstractmethod
    def GetPoints(self,t_array=None):
        pass

    # every child class must provide a method which 
    # takes some subset of its properties to set
    @abstractmethod
    def SetProperties():
        pass

class LineSegment:
    def __init__(self,
            segment_length=50):
        
        super().__init__()

        self._segment_length = segment_length

        self._UpdateCalculatedProperties()

# property getters and setters

    @property
    def segment_length(self):
        return self._segment_length

    @segment_length.setter
    def segment_length(self,new_value):
        self._segment_length = new_value

        self._UpdateCalculatedProperties()

    def SetProperties(self, segment_length):
        self._segment_length = new_value

        self._UpdateCalculatedProperties()

# calculated properties and related functions

    @property
    def final_location(self):
        return self._final_location

    # returns the final rotation vector of the segment
    @property
    def final_orientation(self):
        return [0,0,0]

    def _UpdateCalculatedProperties(self):
        self._UpdateFinalLocation()

    def _UpdateFinalLocation(self):
        # use GetPoints to calculate the final x,y,z point
        self._final_location = self._segment_length

# other functions

    def GetPoints(self,t_array=None):
        assert(len(t_array.shape) == 1)

        # linear interpolation between 0,0,0 and end_point by t
        end_point = np.array([0,0,self._segment_length])
        point_array = end_point * np.expand_dims(t_array,1)

        return point_array

class CircleSegment:
    def __init__(self, 
            segment_length=300, 
            bend_angle=0,
            bend_direction=0):

        super().__init__()

        self._segment_length = segment_length
        self._bend_angle = bend_angle
        self._bend_direction = bend_direction

        self._UpdateCalculatedProperties()

# property getters and setters

    @property
    def segment_length(self):
        return self._segment_length

    @property
    def bend_angle(self):
        return self._bend_angle

    @property
    def bend_direction(self):
        return self._bend_direction

    @bend_angle.setter
    def bend_angle(self, new_value):
        #TODO: Add assert to check new_value in range
        self._bend_angle = new_value

        self._UpdateCalculatedProperties()

    @bend_direction.setter
    def bend_direction(self, new_value):
        #TODO: Add assert to check new_value in range
        self._bend_direction = new_value

        self._UpdateCalculatedProperties()

    def SetProperties(self, bend_angle = None, bend_direction = None):

        if bend_angle is not None:
            self._bend_angle = bend_angle
        if bend_direction is not None:
            self.bend_direction = bend_direction

        self._UpdateCalculatedProperties()

# calculated properties and related functions

    @property
    def final_location(self):
        return self._final_location

    @property
    def final_orientation(self):
        return self._final_orientation
    
    def _UpdateCalculatedProperties(self):
        self._UpdateFinalLocation()
        self._UpdateFinalOrientation()

    def _UpdateFinalLocation(self):
        # use GetPoints to calculate the final x,y,z point
        self._final_location = np.array([0,0,0])

        # uncomment and delete previous line once GetPoints is implemented
        #self._final_location = self.GetPoints(np.array([1]))[0]

    def _UpdateFinalOrientation(self):
        # placeholder function
        self._final_orientation = np.array([0,0,0])

# other functions ig

    def GetPoints(self,t_array=None):
        if t_array is None:
            return np.array([]).reshape((0,3))

        # placeholder return value
        return np.array([]).reshape((0,3))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    line = LineSegment(5)
    arc = CircleSegment(10,1,0.5)

    t_array = np.linspace(0,1,num=5)
    line_points = line.GetPoints(t_array)
    arc_points = arc.GetPoints(t_array)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(line_points[:,0],
            line_points[:,1],
            line_points[:,2],
            color = 'red')
    ax.plot(arc_points[:,0],
            arc_points[:,1],
            arc_points[:,2],
            color = 'blue')

    plt.show()
