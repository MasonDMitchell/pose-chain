import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
from segment2 import LineSegment, CircleSegment
from chain2 import chain2

segment_list.append(CircleSegment,10,1,0.5)

def  cal_cost(theta,X,y):
    '''

    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))

    where:
        j is the no of features
    '''

    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X,y,theta,learning_rate=0.01,cost_history,theta_history):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    #cost_history = np.zeros(iterations)
    #theta_history = np.zeros((iterations,2))


    prediction = np.dot(X,theta)

    theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
    theta_history[,:] =theta.T
    cost_history = cal_cost(theta,X,y)

    return theta, cost_history, theta_history


