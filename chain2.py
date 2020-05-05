from scipy.spatial.transform import Rotation as R
import numpy as np
from segment2 import Segment
import copy

class Chain:
    def __init__(self, 
            segment_count = 5, 
            segment_list = None,
            start_location = None,
            start_orientation = None):

        if segment_list is None:
            self._segment_count = segment_count
            self._segments = [Segment() for idx in range(segment_count)]
        else:
            self._segment_count = len(segment_list)
            assert all([isinstance(segment,Segment) for segment in segment_list])
            self._segments = copy.deepcopy(segment_list)

        if start_location is None:
            start_location = np.array([0,0,0])
        if start_orientation is None:
            start_orientation = R.from_rotvec([0,0,0])

        self._start_location = start_location
        self._start_orientation = start_orientation

        self._UpdateCalculatedProperties()

# property getters and setters

    @property
    def segment_count(self):
        return self._segment_count

    #Generic function for setting properties of any type of segment
    def SetSegmentProperties(idx,*args,**kwargs):
        self._segments[idx].SetProperties(*args,**kwargs)

        self._UpdateCalculatedProperties()

# calculated getters and related functions

    def GetSegments(self):
        return copy.deepcopy(self._segments)

    @property
    def segment_locations(self):
        return self._segment_locations

    @property
    def segment_orientations(self):
        return self._segment_orientations

    def _UpdateCalculatedProperties(self):
        #update orientations must be called before update locations
        self._UpdateSegmentOrientations()
        self._UpdateSegmentLocations()

    def _UpdateSegmentLocations(self):
        segment_orientations = self.segment_orientations
        start_locations = [self._start_location]

        for segment_idx in range(self.segment_count):
            delta_location = segment_orientations[segment_idx].apply(
                self._segments[segment_idx].final_location)
            start_locations.append(
                    start_locations[-1] + delta_location)

        self._segment_locations = start_locations[:-1]

    def _UpdateSegmentOrientations(self):
        orientations = [self._start_orientation]

        for segment_idx in range(self.segment_count):
            delta_orientation = self._segments[segment_idx].final_orientation
            orientations.append(
                    orientations[-1] * delta_orientation)

        self._segment_orientations = orientations[:-1]

# other functions
    
    # t_array is an array of floats from 0 to segment_count
    def GetPoints(self, t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,3))
        assert(len(t_array.shape) == 1)

        point_list = []

        t_array = np.sort(t_array)
        for segment_idx in range(self.segment_count):
            start_orientation = self._segment_orientations[segment_idx]
            start_position = self._segment_locations[segment_idx]

            segment_t = self.__GetSegmentIndices(segment_idx,t_array)

            seg_points = self._segments[segment_idx].GetPoints(segment_t)
            seg_points = start_orientation.apply(seg_points)
            seg_points += start_position

            [point_list.append(point) for point in seg_points]

        points = np.array(point_list)
        return points
        
    def GetOrientations(self, t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,3))
        assert(len(t_array.shape) == 1)

        orientation_list = []

        t_array = np.sort(t_array)
        for segment_idx in range(self.segment_count):
            start_orientation = self._segment_orientations[segment_idx]

            segment_t = self.__GetSegmentIndices(segment_idx,t_array)

            if segment_t.shape[0] == 0:
                continue

            seg_orientations = self._segments[segment_idx].GetOrientations(segment_t)
            seg_orientations = start_orientation * seg_orientations

            [orientation_list.append(quat) for quat in seg_orientations.as_quat()]

        orientations = R.from_quat(orientation_list)

        return orientations

    # find the subset of t_array which is in the specified segment and 
    # map those values to 0-1
    def __GetSegmentIndices(self, segment_idx, t_array):

        #handle end conditions for last segment
        if segment_idx < self.segment_count - 1:
            include_bool = np.logical_and(
                    segment_idx <= t_array,
                    segment_idx + 1 > t_array)
        else:
            include_bool = np.logical_and(
                    segment_idx <= t_array,
                    segment_idx + 1 >= t_array)

        segment_t = t_array[include_bool] # subset of t_array in segment
        segment_t = segment_t - segment_idx # map to 0-1 space

        return segment_t

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from segment2 import LineSegment, CircleSegment
    segment_list = []

    segment_list.append(LineSegment(10))
    segment_list.append(CircleSegment(20,np.pi/4,-np.pi/2))
    segment_list.append(LineSegment(10))
    segment_list.append(CircleSegment(20,np.pi/2,0.))
    segment_list.append(LineSegment(20))

    start_orientation = R.from_rotvec([0,0.2,0])
    start_location = np.array([-5,1,0])

    chain = Chain(segment_list=segment_list,
            start_orientation=start_orientation,
            start_location=start_location)

    t_array = np.linspace(0,5,num=40)

    chain_points = chain.GetPoints(t_array)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(chain_points[:,0],
            chain_points[:,1],
            chain_points[:,2])

    chain_orientations = chain.GetOrientations(t_array)
    tangent = np.array([1,0,0])
    tangent_vecs = chain_orientations.apply(tangent)
    ax.quiver(chain_points[:,0],
            chain_points[:,1],
            chain_points[:,2],
            tangent_vecs[:,0],
            tangent_vecs[:,1],
            tangent_vecs[:,2])

    ax.set_xlim(0,30)
    ax.set_ylim(0,30)
    ax.set_zlim(-30,0)

    print(chain.GetPoints(np.array([5])))

    plt.show()

