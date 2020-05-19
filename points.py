import random
from scipy import spatial
import numpy as np

class Points:
    def __init__(self,chain):

        self.chain = chain
        self.zero_point = [0,0,0]
        self.points = np.array([[0.,0.,0.]]*len(self.chain))
        self.points_vel = np.array([[0.,0.,0.]]*len(self.chain))

        self.bounds = 0
        for i in range(len(self.chain)):
            for j in range(self.chain[i]._segment_count):
                self.bounds += self.chain[i]._segments[j]._segment_length

        self.step = self.bounds/(len(self.points)+1.)

        self.prox_size = self.step/2

        for i in range(len(self.points)):
            self.points[i] = [self.step*(i+1),0.,0.]


    def line_separation(self):
        total_pos_arr = []

        total_pos = np.subtract(self.zero_point[0],self.points[0])
        total_pos += np.subtract(self.points[1],self.points[0])

        length = np.linalg.norm(total_pos)
        if(length==0):
            total_pos_arr.append(np.array([0,0,0]))
        else:
            total_pos_arr.append(total_pos/length)

        for i in range(1,(len(self.points)-1)):
            total_pos = np.subtract(self.points[i-1],self.points[i])
            total_pos += np.subtract(self.points[i+1],self.points[i])
            length = np.linalg.norm(total_pos)
            if(length==0):
                total_pos_arr.append(np.array([0,0,0]))
            else:
                total_pos_arr.append(total_pos/length)

        total_pos = np.subtract(self.points[len(self.points)-2],self.points[len(self.points)-1])
        total_pos += np.subtract(self.points[len(self.points)-3],self.points[len(self.points)-1])

        length = np.linalg.norm(total_pos)
        if(length==0):
            total_pos_arr.append(np.array([0,0,0]))
        else:
            total_pos_arr.append(total_pos/length)


        self.separate = np.array([-1,-1,-1]*np.array(total_pos_arr))

        return self.separate

    def line_cohesion(self):
        avg_pos_arr = []
        avg_pos = (self.points[1]/2)-self.points[0]
        avg_pos_arr.append(avg_pos)
        length = np.linalg.norm(avg_pos_arr[0])
        if length == 0:
            avg_pos_arr[0] = np.array([0,0,0])
        else:
            avg_pos_arr[0] = avg_pos_arr[0]

        for i in range(1,(len(self.points)-1)):
            avg_pos = ((self.points[i-1]+self.points[i+1])/2)-self.points[i]
            avg_pos_arr.append(avg_pos)
            length = np.linalg.norm(avg_pos_arr[i])
            if length ==0:
                avg_pos_arr[i] = np.array([0,0,0])
            else:
                avg_pos_arr[i] = avg_pos_arr[i]

        #avg_pos = ((self.points[len(self.points)-2]+self.points[len(self.points)-3])/2)-self.points[len(self.points)-1]
        #avg_pos_arr.append(avg_pos)

        avg_pos_arr.append(np.array([0,0,0]))
        self.cohesion = avg_pos_arr
        return self.cohesion


    def separation(self):
        tree = spatial.cKDTree(self.points)
        is_close = tree.query_ball_tree(tree,self.prox_size)

        can_see_pos = [self.points[i] for i in is_close]
        total_pos_arr = []

        for i in range(len(self.points)):
            total_pos = np.sum(np.subtract(can_see_pos[i],self.points[i]),axis=0)
            length = np.linalg.norm(total_pos)
            if(length==0):
                total_pos_arr.append(np.array([0,0,0]))
            else:
                total_pos_arr.append(total_pos/length)

        self.separate = np.array([-1,-1,-1]*np.array(total_pos_arr))

        return self.separate

    def cohese(self):
        self.cohesion = []
        for i in range(len(self.points)):
            distance_from = np.subtract(np.array([self.step*(i+1),0,0]),self.points[i])
            length = np.linalg.norm(distance_from)
            if(length == 0):
                self.cohesion.append(np.array([0,0,0]))
            else:
                self.cohesion.append(distance_from/length)

        return self.cohesion

    #Weights is [random,separation,cohesion]
    def update(self,weights):

        self.separation()
        self.cohese()

        for i in range(len(self.points)):
            for j in range(3):
                if self.points_vel[i][j] > 2:
                    self.points_vel[i][j] = 2
                elif self.points_vel[i][j] < -2:
                    self.points_vel[i][j] = -2

            self.points_vel[i] = self.points_vel[i] + self.separate[i]*weights[1] + self.cohesion[i]*weights[2]

            self.points_vel[i] = [self.points_vel[i][0]+random.uniform(-self.bounds*weights[0],self.bounds*weights[0]),self.points_vel[i][1]+random.uniform(-self.bounds*weights[0],self.bounds*weights[0]),self.points_vel[i][2]+random.uniform(-self.bounds*weights[0],self.bounds*weights[0])]

            self.points[i] = self.points[i]+self.points_vel[i]
        return self.points

if __name__ == "__main__":
    from segment2 import ConstLineSegment,CircleSegment
    from chain2 import CompositeSegment, FittingChain
    from scipy.optimize import Bounds
    from matplotlib import pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation
    from matplotlib import animation

    segment_list = []

    segment_list.append(ConstLineSegment(10))
    segment_list.append(CircleSegment(100,0.2,np.pi/2,))

    chain_segments = [CompositeSegment(segment_list=segment_list) for _ in range(5)]

    bounds = Bounds(np.array([-2*np.pi+0.01,-np.inf]*5),
            np.array([2*np.pi-0.01,np.inf]*5),
            keep_feasible=True)

    chain = FittingChain(segment_list=chain_segments,
            bounds=bounds)

    points = Points(chain_segments)


    fig = plt.figure()
    ax = p3.Axes3D(fig)
    scatter, = ax.plot([],[],[],'bo',ms=6)

    def init():
        scatter.set_data([],[])
        scatter.set_3d_properties([])
        return scatter,

    def animate(i):

        goal_points = points.update([.001,.05,.2])
        scatter.set_data(goal_points[:,0],goal_points[:,1])
        scatter.set_3d_properties(goal_points[:,2])
        return scatter,

    ax.set_xlabel("X(mm)")
    ax.set_ylabel("Y(mm)")
    ax.set_zlabel("Z(mm)")
    ax.set_xlim3d([0,500])
    ax.set_ylim3d([-250,250])
    ax.set_zlim3d([-250,250])

    anim = animation.FuncAnimation(fig,animate,init_func=init, interval=20,frames=500)
    #anim.save('videos/fly.mp4',fps=20)
    plt.show()
