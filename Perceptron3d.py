
'''
https://stackoverflow.com/questions/4981815/how-to-remove-lines-in-a-matplotlib-plot
the first response explains objects in matplotlib so well
'''


import matplotlib.pyplot as plt
from random import choice
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import *
from pylab import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time
import mpl_toolkits.mplot3d
import matplotlib
import matplotlib.pyplot as plt

n=100

X=np.random.normal(0.3, 0.2, size=(n, 3))
Y=np.random.uniform(0, 1,n)>0.5
np.random.seed(5)
#X, Y = datasets.make_blobs(n_samples=100,n_features=3, centers=2,cluster_std=1.05, random_state=5)

# y is in {-1, 1}
Y = 2. * Y - 1
X *= Y[:, np.newaxis]
X -= X.mean(axis=0)
X = np.concatenate([np.ones((n, 1)),X], axis=1)



figure, ax = plt.subplots(figsize=(10, 8))
ax = figure.add_subplot(projection='3d')

# setting x-axis label and y-axis label
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


def f(x,y,w):
    return (-w[0]-w[1]*x-w[2]*y)/w[3]


w1=np.array([1,1,1,1])
def loss(n,w,Xi,y):
    somme=0
    
    for i in range(n):
        somme+=np.where(y[i]*np.dot(w, Xi[i, :])<=0,1,0)
    return somme/n  




ax.set_title("Plot different lines with change in a \n f(x)=wx+b", fontsize=15)

ax.scatter3D(X[:,1],X[:,2],X[:,3],c=Y,s=25,alpha=0.3,marker="o")



n_samples = X.shape[0]
n_features = X.shape[1]     
      
w = np.zeros((n_features+1,))
        

'''
x = np.linspace(-3,4,50)

y=np.array([f2d(x,w)]).reshape((1,50))
'''
x=np.array([-1,1])
y=np.array([-1,1])
x,y=np.meshgrid(x,y)
#y = ys.reshape(50,1)
z = f(x,y,w)




t,c=0,0




plot =[ax.plot_surface(x,y,z,alpha=0.45)]

losss=loss(n_samples,w,X,Y)
while losss!=0 :
    losss=loss(n_samples,w,X,Y)
    for i in range(0,n_samples):
        if Y[i]*np.dot(w, X[i, :]) <= 0:
            #ax.set_aspect("auto")
            plt.pause(2)
            plot[0].remove()
            print("w : " ,w,"loss : ",losss , "point mal classifie count : " ,t)
            w += Y[i]*X[i, :]
            z = f(x,y,w)
            
            plot[0]=ax.plot_surface(x,y,z,alpha=0.4)
            t+=1
            losss=loss(n_samples,w,X,Y)
    print( "loss : ", losss)   
plt.pause(1)
plot[0].remove()
z = f(x,y,w)
plot[0]=[ax.plot_surface(x,y,z,alpha=0.35)]
plt.pause(20)
plt.show()


