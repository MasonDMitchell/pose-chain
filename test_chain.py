from tools import createChain
import numpy as np

x = createChain(1,0,0,14)
print(x.GetPoints(np.arange(0,1.5,.5)))
print(x.GetPoints(np.array([.5])))
print(x.GetPoints(np.array([0,1])))


#Bend angle 1 bend direction 1 bend angle 2 bend direction 2
params = np.array([[1],[1],[1],[2]])

x.SetParameters(*params)

print(x.GetParameters())

test = np.array([[1],[1],[2],[2]])

