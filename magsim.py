import numpy as np
import time


class CubeMag:
    def __init__(self,a,b,c,size):

        a = a/1000
        b = b/1000
        c = c/1000

        self.a = np.repeat(a/2,size)
        self.b = np.repeat(b/2,size)
        self.c = np.repeat(c/2,size)
        self.u0 = 4*np.pi*(10**(-7))

        self.Brmax = 1.32
        
        self.M = (self.Brmax)/(self.u0)

        print((self.M*self.u0)/np.pi)

    
    def F1(self,x,y,z):
        XaddA = np.add(x,self.a)
        YaddB = np.add(y,self.b)
        ZaddC = np.add(z,self.c)
        return np.arctan(np.divide(np.multiply(XaddA,YaddB),np.multiply(ZaddC,np.sqrt(np.add(np.add(np.square(XaddA),np.square(YaddB)),np.square(ZaddC))))))

    def F2(self,x,y,z):
        XaddAsq = np.square(np.add(x,self.a))
        ZaddCsq = np.square(np.add(z,self.c))
        return np.divide(np.subtract(np.add(np.sqrt(np.add(np.add(XaddAsq,np.square(np.subtract(y,self.b))),ZaddCsq)),self.b),y),np.subtract(np.subtract(np.sqrt(np.add(np.add(XaddAsq,np.square(np.add(y,self.b))),ZaddCsq)),self.b),y))


    def Bx(self,x,y,z):
        return ((self.u0*self.M)/(4*np.pi))*np.log(np.divide(np.multiply(self.F2(-x,y,-z),self.F2(x,y,z)),np.multiply(self.F2(x,y,-z),self.F2(-x,y,z))))

    def By(self,x,y,z):
        return ((self.u0*self.M)/(4*np.pi))*np.log(np.divide(np.multiply(self.F2(-y,x,-z),self.F2(y,x,z)),np.multiply(self.F2(y,x,-z),self.F2(-y,x,z))))

    def Bz(self,x,y,z):
        return -((self.u0*self.M)/(4*np.pi))*np.add(np.add(np.add(np.add(np.add(np.add(np.add(self.F1(-x,y,z),self.F1(-x,y,-z)),self.F1(-x,-y,z)),self.F1(-x,-y,-z)),self.F1(x,y,z)),self.F1(x,y,-z)),self.F1(x,-y,z)),self.F1(x,-y,-z))

    def getMag(self,x,y,z):
        Bx = self.Bx(x,y,z)
        By = self.By(x,y,z)
        Bz = self.Bz(x,y,z)
        return np.array([Bx,By,Bz])
if __name__ == "__main__":

    x = CubeMag(6.35,6.35,6.35,1)
    y=time.time()
    ans = x.getMag(0,0,.03)*100000
    print(ans)
    ans[0] = ans[0] -30
    ans[1] = ans[1] -35
    ans[2] = ans[2] -35
    z = time.time()

    #print(z-y)
    print(ans)

