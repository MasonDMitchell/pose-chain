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
    '''
    t = np.arange(0,10*np.pi,.01)
    x = (a + b*t) * np.cos(t)
    y = (a + b*t) * np.sin(t)
    '''

    t = np.arange(0,10*np.pi,.01)
    '''
    #Spiral:
    theta=t
    phi=(a+b*t)
    '''
    '''
    #Flower:
    theta=(t/10)**2
    phi=np.sin(t)
    '''
    #Seagull:
    theta=np.sin(t)+np.pi/2
    phi=np.abs(np.sin(t))

    x = phi * np.cos(theta)
    y = phi * np.sin(theta)
    
    ''' 
    plt.style.use('seaborn-paper')
    fig,ax = plt.subplots()
    ax.plot(x,y)
    ax.set_title("Bend Direction & Bend Angle Parameter Space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    plt.show()
    '''
 
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
    phi_max = np.pi/2
    
    phi = params[::2] 
    theta = params[1::2]

    scaled_phi = phi/phi_max

    noise = np.random.normal(0,sigma,(2,len(phi),len(phi[0])))

    vectors = np.multiply(scaled_phi,np.array([np.cos(theta),np.sin(theta)]))

    new_vectors = vectors + noise

    for _ in range(5):
        new_phi = np.linalg.norm(new_vectors,axis=0)
        bad_phi_idxs = np.argwhere(new_phi>1)
        if(bad_phi_idxs.shape[0] == 0):
            break
        noise = np.random.normal(0,sigma,(2,len(bad_phi_idxs),1))
        new_vectors[bad_phi_idxs] = vectors[bad_phi_idxs] + noise

    new_phi = np.linalg.norm(new_vectors,axis=0)
    bad_phi_idxs = np.argwhere(new_phi>1)
    new_vectors[bad_phi_idxs] = vectors[bad_phi_idxs]

    phi = phi_max * np.linalg.norm(new_vectors,axis=0)
    theta = np.arctan2(new_vectors[1],new_vectors[0])

    return np.reshape(zip_lists2(phi,theta),(len(phi)*2,len(phi[0])))

def zip_lists2(a, b):
    return np.ravel(np.column_stack((a,b)))
