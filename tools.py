import matplotlib.pyplot as plt
from chain import CompositeSegment
import numpy as np
from segment import ConstLineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R
from scipy.special import logit,expit

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

def noise(params,sigma):
    alpha = params[0]
    beta = params[1]

    scaled_alpha = alpha/(np.pi)
    epsilon = .000001

    #Create vector
    vector = np.array([np.cos(beta),np.sin(beta)])
    #Alpha scaled to 0 - inf with -x/(x-1)
    big_alpha = -np.divide(scaled_alpha,scaled_alpha-1+epsilon)
    vector = np.multiply(vector,big_alpha)

    #Generate noise and scale with derivative of alpha scale function
    noise = np.random.normal(0,sigma,(2,len(alpha)))
    scaled_noise = np.multiply(np.divide(1,np.square(scaled_alpha-1+epsilon)),noise)

    #Apply scaled noise to vector
    vector = vector + scaled_noise

    #Get length of vector, and then inverse logit
    alpha = np.linalg.norm(vector,axis=0)
    alpha = np.divide(alpha,np.add(1,alpha))
    alpha = alpha*np.pi

    #Get radian direction of vector
    beta = np.arctan2(vector[1],vector[0])

    return np.array([alpha,beta])
