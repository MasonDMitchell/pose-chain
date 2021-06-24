import matplotlib.pyplot as plt
import numpy as np

measured = [[.0189,.02667,.005949],[.1712715,.12827,.20135],[.217977,.26978,.288065],[.196,.279,.333],[.28576,.372,.301],[.574,.334,.313],[.699,.754,.593],[.695,.737,.724],[.824,1.103,1.212],[1.229,1.221,1.395]]
#plt.scatter(np.repeat(np.arange(0,10,1),3),np.reshape(measured,(30,1)))
measured = (np.average(measured,axis=1))
measured = measured - measured[0]
measured = measured * 10
print(measured)

measured2 = [[.4004,.4398,.401],[.5532,.585,.665],[.633,.547,.622],[.697,.6009,.6976],[.7271,.788,.6899],[.8766,.73952,.753879],[1.22699,1.2042,1.1631],[1.086411,1.121,1.1553],[1.814,1.713,1.75],[2.019,2.202,1.89]]
measured_bla = np.average(measured2,axis=1)
#plt.scatter(np.repeat(np.arange(0,10,1),3),np.reshape(measured2-measured_bla[0],(30,1)))
measured2 = np.average(measured2,axis=1)
measured2 = measured2 - measured2[0]
measured2 = measured2 * 10

sim = [0,.3086,.6146,.915,1.2,1.48,1.748,1.9876,2.19819,2.3716]
sim = [0,.8758,1.7438,2.595,3.421,4.21,4.94,5.62,6.207,6.683]

plt.plot(measured,label='measured y')
plt.plot(measured2,label='measured z')
plt.plot(sim,label='sim')
plt.legend()
plt.title("Hall effect readings (zeroed)")
plt.xlabel("Bend angle/10")
plt.ylabel("Magnetic Flux Density")
plt.show()
