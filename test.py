import matplotlib.pyplot as plt
from chain import CompositeSegment
import numpy as np
from segment import ConstLineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R

segment_list = []

segment_list.append(ConstLineSegment(np.array([1])))
segment_list.append(CircleSegment(4,np.array([.25]),np.array([.25])))

chain_segments = [CompositeSegment(segment_list=segment_list) for _ in range(1)]

start_orientation = R.from_rotvec([0,0.2,0])
start_location = np.array([0,0,0])

chain = CompositeSegment(
        segment_list = chain_segments,
        start_orientation = start_orientation,
        start_location = start_location)


print(chain.GetParameters())
