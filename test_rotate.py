from scipy.spatial.transform import Rotation as R
import numpy as np
x = [1,0,0]
print(x)


rotation = np.array([.4,.5,.1])
rotation = rotation / np.linalg.norm(rotation)
print(rotation)
print(np.linalg.norm(rotation))

r = R.from_rotvec(np.pi/2 * rotation)
print(r.as_rotvec())

x = r.apply(x)
print(x)

r = R.from_rotvec(np.pi/2 * (-1*rotation))
print(r.as_rotvec())

x = r.apply(x)
print(x)

