import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


X,y=datasets.make_circles(n_samples=1000,  shuffle=True, noise=0.05, random_state=0, factor=0.3)
for i in range(0,1000):
    if y[i]==0:
        y[i]=-1


def f(x,w):
    if w[1]==0:
        w5=0.001
        return (-w[2]-w[0]*x)/w5
    return (-w[2]-w[0]*x)/w[1]


def loss(n,w,X,y):
    somme=0
    for i in range(0,n):
        somme+=np.where(y[i]*np.dot(w, X[i, :]) <= 0<=0,1,0)
    return somme/n  

        
neg_class = (y == -1)
pos_class = (y == 1)

figure, ax= plt.subplots(figsize=(10, 8))
plt.ion()


Xneg=X[neg_class, 0]

Yneg=X[neg_class, 1]

Xpos=X[pos_class, 0]
Ypos=X[pos_class, 1]

ax.scatter(X[pos_class, 0] ,X[pos_class, 1],color='blue')
ax.scatter(X[neg_class, 0] ,X[neg_class, 1],color='red')
plt.pause(3)
ax.cla() #clear ax


ax.scatter((X[pos_class, 0])**2 ,(X[pos_class, 1])**2,color='blue')
ax.scatter((X[neg_class, 0])**2 ,(X[neg_class, 1])**2,color='red')
ax.axis('scaled') #block x-axis and y-axis 

data = np.power(X,2)


n_samples = data.shape[0]
n=n_samples
n_features = data.shape[1]     
      
w = np.zeros((n_features+1,))

data = np.concatenate([data, np.ones((n_samples, 1))], axis=1)
x = np.linspace(0,4,40)


losss=loss(n,w,data,y)

lines=ax.plot(x, f(x,w))
i,t=0,0
while losss!=0:
    losss=loss(n,w,data,y)
    for i in range(0,n):
        if y[i]*np.dot(w, data[i, :])<=0:
            lines=[]
            plt.pause(0.3)
            ax.lines.pop(0)
            print("w : " ,w,"loss : ",losss , "point mal classifie count : " ,t)
            w=w+y[i]*data[i, :]
            lines=ax.plot(x, f(x,w))
            t+=1

print(loss(n,w,data,y))
plt.pause(0.3)
lines=[]
ax.lines.pop(0)
lines=ax.plot(x, f(x,w))
lines=[]
ax.lines.pop(0)


plt.close()

import numpy as np
from matplotlib import pyplot as plt
from math import pi

u=0     #x-position of the center
v=0     #y-position of the center
a=np.sqrt(abs(w[2]/w[0]))   #radius on the x-axis
b=np.sqrt(abs(w[2]/w[1]))    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.axes()
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) )
plt.scatter(X[pos_class, 0] ,X[pos_class, 1],color='blue')
plt.scatter(X[neg_class, 0] ,X[neg_class, 1],color='red')
plt.pause(8)
plt.show()


        
