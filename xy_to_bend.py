import numpy as np

def convert(x,y,segment_length=3):
    #D is distance to origin
    #Alpha is bend angle
    #Beta is bend direction
    D = np.sqrt(np.add(np.square(x),np.square(y)))
    r = np.repeat(segment_length,len(x))

    alpha = np.arccos(np.subtract(np.repeat(1,len(x)),np.divide(D,r)))
    print(alpha)

    beta = np.arccos(x/D)
    beta = np.multiply(beta,np.divide(y,np.abs(y)))

    return D

if __name__ == "__main__":
    
    x = np.array([1,2])
    y = np.array([1,2])
    convert(x,y)
