import matplotlib.pyplot as plt
import matplotlib as mpl
from chain import CompositeSegment
import numpy as np
from segment import ConstLineSegment, CircleSegment
from scipy.spatial.transform import Rotation as R
from scipy.special import logit,expit
from straight_filter import Filter
#P is # of particles
#N is # of segments
#alpha is bend angle
#beta is bend direction
#S is circle segment length
#L is straight segment length
def createChain(particles,segments,bend_angle,bend_direction,bend_length,straight_length):
    P=particles
    N=segments
    alpha=bend_angle
    beta=bend_direction
    S=bend_length
    L=straight_length

    segment_list = []

    segment_list.append(ConstLineSegment(np.repeat(L,P)))
    segment_list.append(CircleSegment(S,np.repeat(alpha,P),beta))

    chain_segments = [CompositeSegment(segment_list=segment_list) for _ in range(N)]

    start_orientation = R.from_rotvec([0,0,0])
    start_location = np.array([0,0,0])

    chain = CompositeSegment(
            segment_list = chain_segments,
            start_orientation = start_orientation,
            start_location = start_location)
    
    return chain

def spiral_test(segments,bend_angle,bend_direction,bend_length,straight_length):

    #Spiral Parameters
    a = 0
    b = .03
    c = 1
    N=segments
 
    #Create spiral
    #10*pi
    t = np.arange(0,10*np.pi,.1)
    x = (a + b*t) * np.cos(t)
    y = (a + b*t) * np.sin(t)

    """
    plt.style.use('ggplot')
    fig,ax = plt.subplots()
    ax.plot(x,y)
    ax.set_title("Bend Direction & Bend Angle Parameter Space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    a_circle = plt.Circle((0,0),1,fill=False,color='black',linewidth=2)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.add_artist(a_circle)
    plt.show()
    """
 
    #Attain alpha & beta values from x,y values
    vector = np.array([x,y])
    alpha = np.linalg.norm(vector,axis=0)
    beta = np.arctan2(vector[1],vector[0])

    #Create chain and filter with params
    chain = createChain(1,segments,bend_angle,bend_direction,bend_length,straight_length)

    x = Filter(chain,noise)

    #For the purpose of getting through chain
    magnet_array = np.arange(1,N+1,1)
    sensor_array = np.arange(.5,N+.5,1)

    flux = []
    sensor_pos = []
    magnet_pos = []
    for i in range(len(vector[0])):
        params = np.array([[alpha[i]],[beta[i]]])
        params = np.tile(params,(N,1))
        x.chain.SetParameters(*params)
        flux.append(x.compute_flux())
        sensor_pos.append(x.chain.GetPoints(sensor_array))
        magnet_pos.append(x.chain.GetPoints(magnet_array))

    sensor_pos = np.squeeze(sensor_pos)
    magnet_pos = np.squeeze(magnet_pos)

    return flux,sensor_pos,magnet_pos, alpha, beta

def noise(params,sigma):
   
    #For reduction of divide-by-zero errors
    epsilon = .000001 

    alpha = params[::2]
    beta = params[1::2]

    #Scale alpha
    #scaled_alpha = alpha/(np.pi)
    scaled_alpha = alpha/(np.pi/2)

    #Create vector
    vector = np.array([np.cos(beta),np.sin(beta)])

    #Alpha scaled to 0 - inf with -x/(x-1)
    big_alpha = -np.divide(scaled_alpha,scaled_alpha - 1 + epsilon)
    vector = np.multiply(vector,big_alpha)

    #Generate noise and scale with derivative of alpha scale function
    noise = np.random.normal(0,sigma,(2,len(alpha),len(alpha[0])))
    scaled_noise = np.multiply(np.divide(1,np.square(scaled_alpha-1+epsilon)),noise)

    #Apply scaled noise to vector
    vector = vector + scaled_noise

    #Get length of vector, and then inverse logit
    alpha = np.linalg.norm(vector,axis=0)
    alpha = np.divide(alpha,np.add(1,alpha))
    #alpha = alpha*np.pi
    alpha = alpha*(np.pi/2)

    #Get radian direction of vector
    beta = np.arctan2(vector[1],vector[0])

    return np.reshape(zip_lists2(alpha,beta),(len(alpha)*2,len(alpha[0])))

def noise_rejection(params,sigma):
    alpha = params[::2]
    beta = params[1::2]

    #Scale alpha 0-1
    scaled_alpha = alpha/(np.pi/2)

    #Generate unit vector based on orientation
    vector = np.array([np.cos(beta),np.sin(beta)])
    
    #Scale vector based on size of alpha
    vector = np.multiply(vector,scaled_alpha)
    
    noise = np.random.normal(0,sigma,(2,len(alpha),len(alpha[0])))

    new_vector = vector + noise
 
    reject = True
    reject_count = 0
    reject_stop = 4
    while(reject):
        
        #Determine distance from origin
        new_alpha = np.linalg.norm(new_vector,axis=0)

        #Determine vectors that don't fall into the circle
        outside = np.where(new_alpha > 1)[0]
        #print("Reject Count: " + str(reject_count) + " Reject Amount: " + str(len(outside)))
        #Move all remaining vectors back to initial positions after x loops and terminate loop
        reject_count = reject_count + 1
        if(reject_count >= reject_stop):
            new_vector[0][outside] = vector[0][outside]
            new_vector[1][outside] = vector[1][outside]

        #Add new noise to the original positions of the outside vectors
        if(len(outside) > 0 and reject_count < reject_stop):
            noise = np.random.normal(0,sigma,(2,len(alpha),len(alpha[0])))

            new_vector[0][outside] = vector[0][outside] + noise[0]
            new_vector[1][outside] = vector[1][outside] + noise[1]
        else:
            reject = False

    #new_alpha = new_alpha * np.pi
    new_alpha = new_alpha * (np.pi/2)
    new_beta = np.arctan2(new_vector[1],new_vector[0])

    return np.reshape(zip_lists2(new_alpha,new_beta),(len(new_alpha)*2,len(new_alpha[0])))
    #return np.array([new_alpha,new_beta])

def zip_lists(a, b):
    c = np.empty((a.size + b.size), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c
def zip_lists2(a, b):
    return np.ravel(np.column_stack((a,b)))
