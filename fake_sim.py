import chain2
from segment2 import LineSegment, CircleSegment
import numpy as np
import pickle

segment_list = []
circle_segment_length = 100
all_points = []

for k in np.arange(0,np.pi*2,.1):
    for j in range(2):
        if(j==0):
            for i in np.arange(.5,np.pi/2,.05):
                segment_list.append(LineSegment(10))
                segment_list.append(CircleSegment(circle_segment_length,i,k))
                segment_list.append(LineSegment(10))
                segment_list.append(CircleSegment(circle_segment_length,i,k))
                segment_list.append(LineSegment(10))
                segment_list.append(CircleSegment(circle_segment_length,i,k))
                chain = chain2.Chain(segment_list=segment_list)
                all_points.append(chain.GetPoints(np.linspace(0,6,num=60)))
                segment_list = []
        else:
            for i in np.arange(np.pi/2,.5,-.05):
                segment_list.append(LineSegment(10))
                segment_list.append(CircleSegment(circle_segment_length,i,k))
                segment_list.append(LineSegment(10))
                segment_list.append(CircleSegment(circle_segment_length,i,k))
                segment_list.append(LineSegment(10))
                segment_list.append(CircleSegment(circle_segment_length,i,k))
                chain = chain2.Chain(segment_list=segment_list)
                all_points.append(chain.GetPoints(np.linspace(0,6,num=60)))
                segment_list = []

print(len(all_points))
pickle.dump(all_points, open("data/test.p", "wb"))
