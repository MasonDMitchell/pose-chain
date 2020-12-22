from scipy.spatial.transform import Rotation as R
from scipy.optimize import Bounds,minimize
import numpy as np
from segment import AbstractSegment
import copy
import sys

class CompositeSegment(AbstractSegment):
    def __init__(self,
            segment_list,
            start_location = None,
            start_orientation = None):

        self._segment_count = len(segment_list)
        assert all([isinstance(segment,AbstractSegment) for segment in segment_list])
        self._segments = copy.deepcopy(segment_list)
        
        self._instance_count = self._segments[0].GetInstanceCount()
        assert all([self._instance_count == segment.GetInstanceCount() 
                for segment in self._segments])

        if start_location is None:
            start_location = np.array([0,0,0])
        if start_orientation is None:
            start_orientation = R.from_rotvec([0,0,0])

        assert(start_location.shape == (3,))
        assert(start_orientation.as_rotvec().shape == (3,))

        self._start_location = start_location
        self._start_orientation = start_orientation

        self._UpdateCalculatedProperties()

        self._SetParamCount()

# property getters and setters

    @property
    def segment_count(self):
        return self._segment_count

    @property
    def parameter_count(self):
        return self._parameter_count

# calculated getters and related functions

    def GetSegments(self):
        return copy.deepcopy(self._segments)

    def GetInstanceCount(self):
        return self._instance_count

    @property
    def segment_locations(self):
        return self._segment_locations

    @property
    def segment_orientations(self):
        return self._segment_orientations

    @property
    def final_location(self):
        return self._final_location

    @property
    def final_orientation(self):
        return self._final_orientation

    def _UpdateCalculatedProperties(self):
        #update orientations must be called before update locations
        self._UpdateSegmentOrientations()
        self._UpdateSegmentLocations()

    def _UpdateSegmentLocations(self):
        segment_orientations = self.segment_orientations
        start_locations = np.zeros((self._segment_count + 1,self._instance_count,3))
        start_locations[0] = self._start_location[np.newaxis,:]

        for segment_idx in range(self.segment_count):
            delta_location = segment_orientations[segment_idx].apply(
                self._segments[segment_idx].final_location)
            start_locations[segment_idx + 1] = \
                    start_locations[segment_idx] + delta_location

        self._segment_locations = start_locations[:-1]
        self._final_location = start_locations[-1]

    def _GetInstancedStartOrientation(self):
        return R.from_quat([self._start_orientation.as_quat()] * self._instance_count) 

        #segment orientations is segment x instance list of rotations
    def _UpdateSegmentOrientations(self):
        orientations = [self._GetInstancedStartOrientation()]

        for segment_idx in range(self.segment_count):
            delta_orientation = self._segments[segment_idx].final_orientation
            orientations.append(orientations[-1] * delta_orientation)

        self._segment_orientations = orientations[:-1]
        self._final_orientation = orientations[-1]

    def _SetParamCount(self):
        self._parameter_count = sum(segment.parameter_count for segment in self._segments)

# other functions

    # t_array is an array of floats from 0 to segment_count
    def GetPoints(self, t_array = None):
        if t_array is None:
            return np.array([]).reshape((0,self._instance_count,3))
        assert(len(t_array.shape) == 1)

        result = np.zeros((t_array.shape[0], self._instance_count, 3))

        t_array = np.sort(t_array)
        current_idx = 0
        for segment_idx in range(self.segment_count):
            start_orientation = self._segment_orientations[segment_idx]
            start_position = self._segment_locations[segment_idx]

            segment_t = self._GetSegmentIndices(segment_idx,t_array)
            if segment_t.shape[0] == 0:
                continue

            if hasattr(self._segments[segment_idx],"segment_count"):
                segment_t *= self._segments[segment_idx].segment_count
            seg_points = self._segments[segment_idx].GetPoints(segment_t)

            seg_points = np.array([start_orientation.apply(seg_points_t) 
                for seg_points_t in seg_points])
            seg_points += np.expand_dims(start_position,0)

            result[current_idx:current_idx+seg_points.shape[0]] = seg_points
            current_idx += seg_points.shape[0]

        if current_idx < t_array.shape[0]:
            print("Warning: some elements of t_array were outside of the valid range in GetPoints")

        return result

    def GetOrientations(self, t_array = None):
        if t_array is None:
            return []
        assert(len(t_array.shape) == 1)
        if t_array.shape[0] == 0:
            return []

        #orientation_list = [[] for _ in range(self._instance_count)]
        #orientation_list = [[] for _ in range(t_array.shape[0])]
        orientation_list = [R.identity(self._instance_count)] * t_array.shape[0]

        t_array = np.sort(t_array)
        current_idx = 0
        for segment_idx in range(self.segment_count):
            start_orientation = self._segment_orientations[segment_idx]

            segment_t = self._GetSegmentIndices(segment_idx,t_array)

            if segment_t.shape[0] == 0:
                continue

            if hasattr(self._segments[segment_idx],"segment_count"):
                segment_t *= self._segments[segment_idx].segment_count

            seg_orientations = self._segments[segment_idx].GetOrientations(segment_t)

            seg_orientations = [start_orientation * orient_t 
                for orient_t in seg_orientations]

            orientation_list[current_idx:current_idx+len(seg_orientations)] \
                = seg_orientations
            current_idx += len(seg_orientations)

        if current_idx < t_array.shape[0]:
            print("Warning: some elements of t_array were outside of the valid range in GetOrientations")

        return orientation_list

    # find the subset of t_array which is in the specified segment and
    # map those values to 0-1
    def _GetSegmentIndices(self, segment_idx, t_array):

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

    def SetParameters(self,*params):
        total_param_count = 0

        for segment in self._segments:
            param_count = segment.parameter_count

            segment_params = params[total_param_count:total_param_count + param_count]
            total_param_count += param_count
                
            segment.SetParameters(*segment_params)

        instance_count = self._segments[0].GetInstanceCount()
        assert(all([segment.GetInstanceCount() == instance_count 
            for segment in self._segments]))
        self._instance_count = instance_count

        self._UpdateCalculatedProperties()

    def GetParameters(self):
        params = []
        for segment in self._segments:
            params += list(segment.GetParameters())
        return np.array(params)

    #Generic function for setting properties of any type of segment
    def SetSegmentProperties(idx,*args,**kwargs):
        self._segments[idx].SetProperties(*args, **kwargs)
        
        assert(self._segments[idx].GetInstanceCount() == self._instance_count)

        self._UpdateCalculatedProperties()

#Alias Chain to CompositeSegment
Chain = CompositeSegment

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from segment import ConstLineSegment, CircleSegment
    segment_list = []

    segment_list.append(ConstLineSegment(np.array([4,5,7,1])))
    segment_list.append(CircleSegment(4,np.array([-0.25,0.5,2,0.25]),np.array([0,0.5,0,-1])))

    chain_segments = [CompositeSegment(segment_list=segment_list) for _ in range(4)]

    start_orientation = R.from_rotvec([0,0.2,0])
    start_location = np.array([0,0,0])

    chain = Chain(
            segment_list = chain_segments,
            start_orientation = start_orientation,
            start_location = start_location)


    #chain = FittingChain(segment_list=chain_segments,
    #        start_orientation=start_orientation,
    #        start_location=start_location,
    #        points=goal_points,
    #        bounds=bounds)

    t_array = np.linspace(0,4,num=41)

    chain_points = chain.GetPoints(t_array)

    print(chain_points.shape)

    chain_orientations = chain.GetOrientations(t_array)
    tangent = np.array([0.5,0,0])
    tangent_vecs = np.array([orient_t.apply(tangent)
            for orient_t in chain_orientations])
    print(tangent_vecs.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(4):
        ax.plot(chain_points[:,idx,0],
                chain_points[:,idx,1],
                chain_points[:,idx,2])

        ax.quiver(chain_points[:,idx,0],
                chain_points[:,idx,1],
                chain_points[:,idx,2],
                tangent_vecs[:,idx,0],
                tangent_vecs[:,idx,1],
                tangent_vecs[:,idx,2])

    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_zlim(-10,0)

    print(chain.GetPoints(np.array([4])))
    plt.show()
