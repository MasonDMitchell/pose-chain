import matplotlib.pyplot as plt
from chain import CompositeSegment
import numpy as np
from segment import ConstLineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R

#P is # of particles
#N is # of segments
#alpha is bend angle
#beta is bend direction
#S is circle segment length
#L is straight segment length
def createChain(P,N,alpha,beta,S,L):
    segments = []

    segments.append(ConstLineSegment(np.repeat(L,P)))
    segments.append(CircleSegment(S,np.repeat(alpha,P),beta))

    chain_segments = [CompositeSegment(segment_list=segments) for _ in range(N)]

    start_orientation = R.from_rotvec([0,0,0])
    start_location = np.array([0,0,0])

    chain = CompositeSegment(
            segment_list = chain_segments,
            start_orientation = start_orientation,
            start_location = start_location)
    
    return chain
