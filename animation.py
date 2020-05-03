import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
class Animate:
    def __init__(self,all_points):
        self.all_points = all_points
        self.lim = [[-100,100],[-100,100],[-100,100]]
        self.rotate = False
        self.show = False
        self.save = False
        self.filename = "unnamed"
        self.dpi = 50
        plt.style.use('ggplot')

        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        self.line, = self.ax.plot([],[],[],color='black')


    def SetParameters(self,dpi = None, filename = None, show=None, save=None, rotate=None,lim = None):
        if save is not None:
            self.save = save
        if show is not None:
            self.show = show
        if rotate is not None:
            self.rotate = rotate
        if lim is not None:
            self.lim = lim
        if filename is not None:
            self.filename = filename
        if dpi is not None:
            self.dpi = dpi
        return
    def _PlotInit(self):
        self.line.set_data([],[])
        self.line.set_3d_properties([])
        return self.line,

    def _PlotAnimate(self,i):
        if self.rotate is True:
            self.ax.view_init(azim=130+(i*.3))
        self.line.set_data(self.all_points[i][:,0],self.all_points[i][:,1])
        self.line.set_3d_properties(self.all_points[i][:,2])
        return self.line,
    
    def Plot(self):

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")

        self.ax.set_xlim3d(self.lim[0])
        self.ax.set_ylim3d(self.lim[1])
        self.ax.set_zlim3d(self.lim[2])

        self.ax.set_title("Proprioceptive Chain")

        blitting = True
        if self.show is True and self.rotate is True:
            blitting = False
        anim = animation.FuncAnimation(self.fig, self._PlotAnimate, init_func=self._PlotInit, frames = len(all_points), interval=20, blit = blitting)
        
        if self.show is True:
            plt.show()

        if self.save is True:
            anim.save('videos/' + str(self.filename) + '.mp4', fps=80, bitrate= (self.dpi*8), dpi= self.dpi, extra_args=['-vcodec','libx264'])

if __name__ == "__main__":
    import pickle
    all_points = pickle.load(open("data/test.p","rb"))
    plot = Animate(all_points)
    plot.SetParameters(save=True,rotate=True,dpi=100)
    plot.Plot()
