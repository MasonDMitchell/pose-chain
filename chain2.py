from scipy.spatial.transform import Rotation as R
from scipy.optimize import Bounds,minimize
import numpy as np
from segment2 import AbstractSegment
import copy

class CompositeSegment(AbstractSegment):
    def __init__(self, 
            segment_list,
            start_location = None,
            start_orientation = None):

        self._segment_count = len(segment_list)
        assert all([isinstance(segment,AbstractSegment) for segment in segment_list])
        self._segments = copy.deepcopy(segment_list)

        if start_location is None:
            start_location = np.array([0,0,0])
        if start_orientation is None:
            start_orientation = R.from_rotvec([0,0,0])

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
        start_locations = [self._start_location]

        for segment_idx in range(self.segment_count):
            delta_location = segment_orientations[segment_idx].apply(
                self._segments[segment_idx].final_location)
            start_locations.append(
                    start_locations[-1] + delta_location)

        self._segment_locations = start_locations[:-1]
        self._final_location = start_locations[-1]

    def _UpdateSegmentOrientations(self):
        orientations = [self._start_orientation]

        for segment_idx in range(self.segment_count):
            delta_orientation = self._segments[segment_idx].final_orientation
            orientations.append(
                    orientations[-1] * delta_orientation)

        self._segment_orientations = orientations[:-1]
        self._final_orientation = orientations[-1]

    def _SetParamCount(self):
        self._parameter_count = sum(segment.parameter_count for segment in self._segments)

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

            segment_t = self._GetSegmentIndices(segment_idx,t_array)

            if segment_t.shape[0] == 0:
                continue

            if hasattr(self._segments[segment_idx],"segment_count"):
                segment_t *= self._segments[segment_idx].segment_count

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

            segment_t = self._GetSegmentIndices(segment_idx,t_array)

            if segment_t.shape[0] == 0:
                continue

            if hasattr(self._segments[segment_idx],"segment_count"):
                segment_t *= self._segments[segment_idx].segment_count

            seg_orientations = self._segments[segment_idx].GetOrientations(segment_t)
            seg_orientations = start_orientation * seg_orientations

            [orientation_list.append(quat) for quat in seg_orientations.as_quat()]

        orientations = R.from_quat(orientation_list)

        return orientations

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
        
        self._UpdateCalculatedProperties()

    def GetParameters(self):
        params = []
        for segment in self._segments:
            params += segment.GetParameters()
        return params

    #Generic function for setting properties of any type of segment
    def SetSegmentProperties(idx,*args,**kwargs):
        self._segments[idx].SetProperties(*args, **kwargs)

        self._UpdateCalculatedProperties()

#Alias Chain to CompositeSegment
Chain = CompositeSegment

class FittingChain(AbstractSegment):
    def __init__(self,
            *args,
            initial_parameters=None,
            bounds=None,
            points=None,
            error='sqeuclidean',
            method='TNC',
            **kwargs):
        self._chain = CompositeSegment(*args, **kwargs)

        if initial_parameters is None:
            initial_parameters = np.array(self._chain.GetParameters())

        self._current_params = initial_parameters

        if bounds is None:
            bounds = Bounds(np.array([-np.inf] * self._chain.parameter_count),
                    np.array([np.inf] * self._chain.parameter_count),
                    keep_feasible=True)

        assert(bounds.lb.shape[0] == self._chain.parameter_count)
        assert(bounds.ub.shape[0] == self._chain.parameter_count)
        self._bounds = bounds

        if error == 'sqeuclidean':
            self._error_func = self.SumSquaredEuclidean

        self._method = method

        if points is not None:
            self.SetFittingPoints(points)

    @property
    def parameter_count(self):
        return 3 * self._chain.segment_count

    @property
    def final_location(self):
        return self._chain.final_location

    @property
    def final_orientation(self):
        return self._chain.final_orientation

    def GetPoints(self, t_array=None):
        return self._chain.GetPoints(t_array)

    def GetOrientations(self, t_array=None):
        return self._chain.GetOrientations(t_array)

    def SetParameters(self, *points):
        points = np.array(points).reshape((-1,3))
        self.SetFittingPoints(points)

    def GetParameters(self):
        return list(self._points.flatten())

    def SetFittingPoints(self, points):
        
        testing_chain = copy.deepcopy(self._chain)
        def eval_foo(parameters):
            return self._EvaluateChainParameters(testing_chain, 
                    self._error_func, 
                    parameters, 
                    points)

        res = minimize(fun=eval_foo,
                x0=self._current_params,
                method=self._method,
                bounds=self._bounds)

        self._current_params = res.x
        self._chain.SetParameters(*res.x)
        self._points = points

    @staticmethod
    def _EvaluateChainParameters(composite_segment, error_func, parameters, goal_points):
        
        point_count = goal_points.shape[0]
        
        composite_segment.SetParameters(*parameters)

        t_array = np.arange(start=1,stop=point_count + 1,step=1)
        chain_points = composite_segment.GetPoints(t_array)

        error = error_func(chain_points,goal_points)

        return error

    @staticmethod
    def SumSquaredEuclidean(current, goal):
        current = np.array(current)
        goal = np.array(goal)

        difference_sq = np.power(current - goal, 2)
        all_sum = np.sum(difference_sq)

        return all_sum


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from segment2 import ConstLineSegment, CircleSegment
    segment_list = []

    segment_list.append(ConstLineSegment(2))
    segment_list.append(CircleSegment(4,0.01,0.))

    chain_segments = [CompositeSegment(segment_list=segment_list) for _ in range(5)]

    start_orientation = R.from_rotvec([0,0.2,0])
    start_location = np.array([0,0,0])

    goal_points = np.array(
            [[5,1,0],
            [10,0,0],
            [12,4,0],
            [12,4,6],
            [12,5,3]])

    bounds = Bounds(np.array([0,-np.inf] * 5),
            np.array([2*np.pi,np.inf] * 5),
            keep_feasible=True)

    chain = FittingChain(segment_list=chain_segments,
            start_orientation=start_orientation,
            start_location=start_location,
            points=goal_points,
            bounds=bounds)

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

    ax.scatter(goal_points[:,0],
            goal_points[:,1],
            goal_points[:,2],
            color='red')

    ax.set_xlim(0,30)
    ax.set_ylim(0,30)
    ax.set_zlim(-30,0)

    print(chain.GetPoints(np.array([5])))
    plt.show()

