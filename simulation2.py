from segment2 import ConstLineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R
from chain2 import CompositeSegment, FittingChain
from points import Points
import numpy as np
from scipy.optimize import Bounds
import pickle
segment_list = []

segment_list.append(ConstLineSegment(10))
segment_list.append(CircleSegment(100,0.2,np.pi/2.))

chain_segments = [CompositeSegment(segment_list=segment_list) for _ in range(5)]

start_orientation = R.from_rotvec([0,0.2,0])
start_location = np.array([0,0,0])

bounds = Bounds(np.array([0,-np.inf] * 5),
            np.array([np.pi/2,np.inf] * 5),
            keep_feasible=True)


all_points = []

points = Points(chain_segments)
for i in range(50):
    print(i)
#need to generate new goal_points
    goal_points = points.update([.001,.05,.2])
    chain = FittingChain(segment_list=chain_segments,
            start_orientation=start_orientation,
            start_location=start_location,
            points=goal_points,
            bounds=bounds)

    t_array = np.linspace(0,5,num=500)

    all_points.append(chain.GetPoints(t_array))

pickle.dump(all_points,open('data/simPickle.p','ab'))

