from abc import ABCMeta, abstractmethod, abstractproperty
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
from numbers import Number

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

    # returns the number of instances of the segment held by this object
    def GetInstanceCount(self):
        pass

class LineSegment(AbstractSegment):
    def __init__(self,
            segment_length=None):

        super().__init__()

        if segment_length is None:
            segment_length = np.array([50])
        else:
            if isinstance(segment_length, Number):
                segment_length = np.array([segment_length])
            else:
                segment_length = np.array(segment_length)

        assert(len(segment_length.shape) == 1)

        self._segment_length = segment_length
        self._instance_count = segment_length.shape[0]

        self._UpdateCalculatedProperties()

# property getters and setters

    @property
    def segment_length(self):
        return self._segment_length

    @segment_length.setter
    def segment_length(self,new_value):
        if isinstance(new_value, Number):
            new_value = np.array([new_value])
        else:
            new_value = np.array(new_value)

        new_value = np.array(new_value)
        assert(len(new_value.shape) == 1)

        self._segment_length = new_value
        self._instance_count = new_value.shape[0]

        self._UpdateCalculatedProperties()

    @property
    def parameter_count(self):
        return 1

    def SetParameters(self, segment_length):
        assert(len(new_value.shape) == 1)

        self._segment_length = segment_length
        self._instance_count = segment_length.shape[1]

        self._UpdateCalculatedProperties()

    def GetParameters(self):
        return np.array([self._segment_length])

    def GetInstanceCount(self):
        return self._instance_count

# calculated properties and related functions

    @property
    def final_location(self):
        return self._final_location

    # returns the final rotation vector of the segment
    @property
    def final_orientation(self):
        return R.identity(self._instance_count)

    def _UpdateCalculatedProperties(self):
        self._UpdateFinalLocation()

    def _UpdateFinalLocation(self):
        final_location = np.zeros((self._instance_count,3))
        final_location[:,0] = self._segment_length

        self._final_location = final_location

# other functions

    def GetPoints(self,t_array=None):
        if t_array is None:
            return np.array([]).reshape((0,0,3))
        assert(len(t_array.shape) == 1)
        t_array = np.expand_dims(t_array,1)

        # linear interpolation between 0,0,0 and end_point by t
        end_point = np.zeros((self._instance_count,3))
        end_point[:,0] = self._segment_length

        point_array = np.expand_dims(end_point,0) * np.expand_dims(t_array,2)

        return point_array

    def GetOrientations(self, t_array = None):
        if t_array is None:
            return []
        assert(len(t_array.shape) == 1)
        if t_array.shape[0] == 0:
            return []

        return [R.identity(self._instance_count)] * t_array.shape[0]

class ConstLineSegment(LineSegment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    @property
    def parameter_count(self):
        return 0

    def SetParameters(self):
        pass

    def GetParameters(self):
        return np.array([]).reshape((0, self.GetInstanceCount()))

# segment_length, bend_angle, and bend_direction can be either scalar or 
# vector. Currently CircleSegment only supports fixed lengths after creation. 
# This has other implications, for example, if the initial parameters are 
# vector valued with multiple elements, then the number of instances held by 
# the segment cannot be changed later. However if the initial number of 
# instances is 0 then when the number of instances is changed later by 
# setting the parameters, all instances are assumed to have the same length.
class CircleSegment(AbstractSegment):
    def __init__(self,
            segment_length=None,
            bend_angle=None,
            bend_direction=None):

        super().__init__()

        instance_count = 1
        if segment_length is not None:
            if not isinstance(segment_length, Number):
                segment_length = np.array(segment_length)
                assert(len(segment_length.shape) == 1)
                instance_count = segment_length.shape[0]

        if bend_angle is not None:
            if not isinstance(bend_angle, Number):
                bend_angle = np.array(bend_angle)
                assert(len(bend_angle.shape) == 1)
                instance_count = bend_angle.shape[0]

        if bend_direction is not None:
            if not isinstance(bend_direction, Number):
                bend_direction = np.array(bend_direction)
                assert(len(bend_direction.shape) == 1)
                instance_count = bend_direction.shape[0]

        if isinstance(segment_length, Number):
            segment_length = np.array([segment_length] * instance_count)
        if isinstance(bend_angle, Number):
            bend_angle = np.array([bend_angle] * instance_count)
        if isinstance(bend_direction, Number):
            bend_direction = np.array([bend_direction] * instance_count)

        if segment_length is None:
            segment_length = np.array([100] * instance_count)
        if bend_angle is None:
            bend_angle = np.array([0] * instance_count)
        if bend_direction is None:
            bend_direction = np.array([0] * instance_count)
        
        assert(len(segment_length.shape) == 1)
        assert(len(bend_angle.shape) == 1)
        assert(len(bend_direction.shape) == 1)

        assert(segment_length.shape == bend_angle.shape)
        assert(bend_angle.shape == bend_direction.shape)

        self._segment_length = segment_length
        self._bend_angle = bend_angle
        self._bend_direction = bend_direction

        self._instance_count_changeable=False
        if instance_count == 1:
            self._instance_count_changeable = True
        self._instance_count = instance_count

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
        if isinstance(new_value, Number):
            new_value = np.array([new_value])
        else:
            new_value = np.array(new_value)

        assert(len(new_value.shape) == 1)
        assert(new_value.shape[0] == self._instance_count)
        assert(np.all(-2 * np.pi <= new_value) and np.all(new_value <= 2 * np.pi))

        #self._bend_direction = np.where(
        #    self._bend_angle * new_value < 0,
        #    self._bend_direction + np.pi,
        #    self._bend_direction)    

        self._bend_angle = new_value

        self._UpdateCalculatedProperties()

    @bend_direction.setter
    def bend_direction(self, new_value):
        if isinstance(new_value, Number):
            new_value = np.array([new_value])
        else:
            new_value = np.array(new_value)

        assert(len(new_value.shape) == 1)
        assert(new_value.shape[0] == self._instance_count)
        self._bend_direction = new_value

        self._UpdateCalculatedProperties()

    def SetParameters(self, bend_angle = None, bend_direction = None):
        
        instance_count = self._instance_count
        if bend_angle is not None:
            assert(len(bend_angle.shape) == 1)
            instance_count = bend_angle.shape[0]
        elif bend_direction is not None:
            assert(len(bend_direction.shape) == 1)
            instance_count = bend_direction.shape[0]

        if not self._instance_count_changeable:
            assert(instance_count == self._instance_count)
        else:
            self._segment_length = np.full(
                    (self._instance_count),
                    self._segment_length[0])
        
        assert(bend_angle.shape == bend_direction.shape)
        assert(-2 * np.pi <= np.all(bend_angle) and np.all(bend_angle) <= 2 * np.pi)

        if bend_direction is not None:
            self._bend_direction = bend_direction

        if bend_angle is not None:
            self._bend_angle = bend_angle

        self._UpdateCalculatedProperties()

    def GetParameters(self):
        return np.array([self._bend_angle, self._bend_direction])

    def GetInstanceCount(self):
        return self._instance_count

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
        self._radius = np.where(
            np.isclose(self.bend_angle,0),
            1e6,
            self._segment_length / np.abs(self.bend_angle))

# other functions ig
    def _AdjustedDirection(self):
        adjusted_direction = np.where(
                self.bend_angle < 0,
                self._bend_direction + np.pi,
                self._bend_direction)

        return adjusted_direction

    def GetPoints(self,t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,0,3))
        assert(len(t_array.shape) == 1)
        t_array = np.expand_dims(t_array,1)

        abs_angle = np.expand_dims(np.abs(self.bend_angle),axis=0)
        radius = np.expand_dims(self.radius,axis=0)
        horizontal_dist = radius - radius * np.cos(t_array * abs_angle)
        vertical_dist = radius * np.sin(t_array * abs_angle)

        vertical_dist = np.where(
            np.isclose(abs_angle, 0),
            t_array * np.expand_dims(self._segment_length,axis=0),
            vertical_dist)

        adjusted_direction = np.expand_dims(
            self._AdjustedDirection(),
            axis=0)

        points_y = np.cos(adjusted_direction) * horizontal_dist
        points_z = np.sin(adjusted_direction) * horizontal_dist

        points = np.stack([vertical_dist,points_y,points_z],axis=2)

        return points

    def GetOrientations(self, t_array = None):
        if t_array is None:
            return []
        assert(len(t_array.shape) == 1)
        if t_array.shape[0] == 0:
            return []
        t_array = np.expand_dims(t_array,1)

        abs_angle = np.expand_dims(np.abs(self.bend_angle),axis=0)
        adjusted_direction = np.expand_dims(
            self._AdjustedDirection(),
            axis=0)

        norm_y = -np.sin(adjusted_direction)
        norm_z = np.cos(adjusted_direction)
        length = t_array * abs_angle

        vec_y = norm_y * length
        vec_z = norm_z * length

        rotvec = np.stack(
            [np.zeros((t_array.shape[0],self._instance_count)), vec_y, vec_z],
            axis=2)

        orientations = [R.from_rotvec(rotvec_t) for rotvec_t in rotvec]

        return orientations

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    line = LineSegment(np.array([20,30]))
    arc = CircleSegment(35,
            np.array([2,1,0.,-1]),
            np.array([1.5*np.pi,0.,1.,0.]))

    t_array = np.linspace(0,1,num=5)
    line_points = line.GetPoints(t_array)
    arc_points = arc.GetPoints(t_array)

    line_orientations = line.GetOrientations(t_array)
    arc_orientations = arc.GetOrientations(t_array)

    line.GetOrientations()
    arc.GetOrientations()

    vec_lengths = 4
    tangent = np.array([vec_lengths,0,0])
    normal = np.array([0,vec_lengths,0])

    tangent_vecs = np.array([arc_t.apply(tangent) for arc_t in arc_orientations])
    normal_vecs = np.array([arc_t.apply(normal) for arc_t in arc_orientations])
    #print("Printing Output")
    #print(tangent_vecs)
    #print(normal_vecs)

    #print(line_points)
    #print([x.as_euler('xyz') for x in line_orientations])
    #print(arc_points)
    #print([x.as_euler('xyz') for x in arc_orientations])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(line_points[0,:,0],
            line_points[0,:,1],
            line_points[0,:,2],
            color = 'red')

    for idx in range(4):

        ax.plot(arc_points[:,idx,0],
                arc_points[:,idx,1],
                arc_points[:,idx,2],
                color = 'blue')

        ax.quiver(arc_points[:,idx,0],
                arc_points[:,idx,1],
                arc_points[:,idx,2],
                tangent_vecs[:,idx,0],
                tangent_vecs[:,idx,1],
                tangent_vecs[:,idx,2])

        ax.quiver(arc_points[:,idx,0],
                arc_points[:,idx,1],
                arc_points[:,idx,2],
                normal_vecs[:,idx,0],
                normal_vecs[:,idx,1],
                normal_vecs[:,idx,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-35,35)
    ax.set_ylim(-35,35)
    ax.set_zlim(-35,35)

    plt.show()
