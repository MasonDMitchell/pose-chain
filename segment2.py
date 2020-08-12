from abc import ABCMeta, abstractmethod, abstractproperty
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

class AbstractSegment(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractproperty
    def final_location(self):
        pass

    # returns the final rotation vector of the segment
    @abstractproperty
    def final_orientation(self):
        pass

    # number of arguments in SetProperties
    @abstractproperty
    def parameter_count(self):
        pass

    # every child class must provide a method which 
    # takes some subset of its properties to set
    @abstractmethod
    def SetParameters(self):
        pass

    @abstractmethod
    def GetParameters(self):
        pass

    # returns x,y,z coordinates for each t in t_array
    # t varies from 0 to 1
    # t_array can contain a single element
    @abstractmethod
    def GetPoints(self,t_array=None):
        pass

    # returns a scipy rotation to describe all orientations in t_array
    @abstractmethod
    def GetOrientations(self,t_array=None):
        pass

class LineSegment(AbstractSegment):
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

    @property
    def parameter_count(self):
        return 1

    def SetParameters(self, segment_length):
        self._segment_length = new_value

        self._UpdateCalculatedProperties()

    def GetParameters(self):
        return [self._segment_length]

# calculated properties and related functions

    @property
    def final_location(self):
        return self._final_location

    # returns the final rotation vector of the segment
    @property
    def final_orientation(self):
        return R.from_rotvec([0,0,0])

    def _UpdateCalculatedProperties(self):
        self._UpdateFinalLocation()

    def _UpdateFinalLocation(self):
        self._final_location = np.array([self._segment_length,0,0])

# other functions

    def GetPoints(self,t_array=None):
        if t_array is None:
            return np.array([]).reshape((0,3))
        assert(len(t_array.shape) == 1)

        # linear interpolation between 0,0,0 and end_point by t
        end_point = np.array([self._segment_length,0,0])
        point_array = end_point * np.expand_dims(t_array,1)

        return point_array

    def GetOrientations(self, t_array = None):
        if t_array is None:
            return R.from_rotvec(np.array([]).reshape((0,3)))
        assert(len(t_array.shape) == 1)
        if t_array.shape[0] == 0:
            return R.from_rotvec(np.array([]).reshape((0,3)))

        return R.from_rotvec([[0,0,0]] * t_array.shape[0])

class ConstLineSegment(LineSegment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    @property
    def parameter_count(self):
        return 0

    def SetParameters(self):
        pass

    def GetParameters(self):
        return []

class CircleSegment(AbstractSegment):
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
    def parameter_count(self):
        return 2

    @property
    def bend_angle(self):
        return self._bend_angle

    @property
    def bend_direction(self):
        return self._bend_direction

    @bend_angle.setter
    def bend_angle(self, new_value):
        assert(-2 * np.pi <= new_value and new_value <= 2 * np.pi)

        if self._bend_angle * new_value < 0:
            self._bend_direction += np.pi

        self._bend_angle = abs(new_value)

        self._UpdateCalculatedProperties()

    @bend_direction.setter
    def bend_direction(self, new_value):
        self._bend_direction = new_value

        self._UpdateCalculatedProperties()

    def SetParameters(self, bend_angle = None, bend_direction = None):
        assert(-2 * np.pi <= bend_angle and bend_angle <= 2 * np.pi)

        if bend_direction is not None:
            self._bend_direction = bend_direction
            if bend_angle is not None:
                if bend_angle < 0:
                    self._bend_direction += np.pi
            else:
                if self._bend_angle < 0:
                    self._bend_direction += np.pi
        else:
            if bend_angle is not None:
                if self._bend_angle * bend_angle < 0:
                    self._bend_direction += np.pi

        if bend_angle is not None:
            self._bend_angle = bend_angle

        self._UpdateCalculatedProperties()

    def GetParameters(self):
        return [self._bend_angle, self._bend_direction]

# calculated properties and related functions

    @property
    def final_location(self):
        return self._final_location

    @property
    def final_orientation(self):
        return self._final_orientation

    @property
    def radius(self):
        return self._radius

    def _UpdateCalculatedProperties(self):
        self._UpdateRadius()
        self._UpdateFinalLocation()
        self._UpdateFinalOrientation()

    def _UpdateFinalLocation(self):
        self._final_location = self.GetPoints(np.array([1]))[0]

    def _UpdateFinalOrientation(self):
        self._final_orientation = self.GetOrientations(np.array([1]))[0]

    def _UpdateRadius(self):
        if math.isclose(self.bend_angle,0):
            self._radius = 1e6
        else:
            self._radius = self.segment_length / self.bend_angle


# other functions ig

    def GetPoints(self,t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,3))
        assert(len(t_array.shape) == 1)

        horizontal_dist = self.radius - self.radius * np.cos(t_array * self.bend_angle)
        vertical_dist = self.radius * np.sin(t_array * self.bend_angle)

        points_y = np.cos(self.bend_direction) * horizontal_dist
        points_z = np.sin(self.bend_direction) * horizontal_dist

        points = np.stack([vertical_dist,points_y,points_z],axis=1)

        # placeholder return value
        return points

    def GetOrientations(self, t_array = None):
        if t_array is None:
            return R.from_rotvec(np.array([]).reshape((0,3)))
        assert(len(t_array.shape) == 1)
        if t_array.shape[0] == 0:
            return R.from_rotvec(np.array([]).reshape((0,3)))

        norm_y = -np.sin(self.bend_direction)
        norm_z = np.cos(self.bend_direction)
        length = t_array * self.bend_angle

        rotvec = np.expand_dims(length,axis=1) * np.expand_dims(np.array([0,norm_y,norm_z]),axis=0)

        orientations = R.from_rotvec(rotvec)

        '''
        #equivalent to previous code, but using euler angles
        pitch = t_array * self.bend_angle
        yaw = -t_array * self.bend_direction
        angles = [[0,pitch_val,yaw_val] for pitch_val, yaw_val in zip(pitch,yaw)]
        orientations = R.from_euler('xzy',angles)
        '''

        return orientations

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    line = LineSegment(5)
    arc = CircleSegment(35,.001,1.5*np.pi)

    t_array = np.linspace(0,1,num=5)
    line_points = line.GetPoints(t_array)
    arc_points = arc.GetPoints(t_array)

    line_orientations = line.GetOrientations(t_array)
    arc_orientations = arc.GetOrientations(t_array)

    line.GetOrientations()
    arc.GetOrientations()

    tangent = np.array([1,0,0])
    normal = np.array([0,1,0])

    tangent_vecs = arc_orientations.apply(tangent)
    normal_vecs = arc_orientations.apply(normal)
    print("Printing Output")
    print(tangent_vecs)
    print(normal_vecs)

    print(line_points)
    print(line_orientations.as_euler('xyz'))
    print(arc_points)
    print(arc_orientations.as_euler('xyz'))

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

    print(arc_points[:,0])
    print(tangent_vecs[:,0])
    ax.quiver(arc_points[:,0],
            arc_points[:,1],
            arc_points[:,2],
            tangent_vecs[:,0],
            tangent_vecs[:,1],
            tangent_vecs[:,2])

    ax.quiver(arc_points[:,0],
            arc_points[:,1],
            arc_points[:,2],
            normal_vecs[:,0],
            normal_vecs[:,1],
            normal_vecs[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)

    plt.show()
