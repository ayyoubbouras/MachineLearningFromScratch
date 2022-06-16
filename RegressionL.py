import pandas as pd
import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.optimize import minimize_scalar

fig = plt.figure()
df=pd.read_csv('RL.csv')
x=np.array(df['X'])
y=np.array(df['Y'])
plt.pause(1)
fig.clear()
m=len(x)
def grad_loss(w):
    S=0
    for i,j in zip(x,y):
        S+=(np.dot(np.array([w[0],w[1]]),np.array([1,i]))-j)*np.array([1,i])
    return (2*S)/m
def loss(w):
    S=0
    for i,j in zip(x,y):
        S+=(np.dot(np.array([w[0],w[1]]),np.array([1,i]))-j)**2
    return S/m
def alphaSearch(w):
  grad = grad_loss(w)
  a = sp.Symbol("a")
  wk = w - a*grad
  f = str(loss(wk))
  def fun(a):
    fu = eval(f)
    return fu
  alpha = minimize_scalar(fun)
  return alpha.x
w=np.array([1,-5])
gls=grad_loss(w)
ls=loss(w)
plt.scatter(x,y,s=0.5)
while la.norm(gls) > 0.0001:
    alpha=alphaSearch(w)
    print(alpha)
    w=w-alpha*gls
    a = np.array([np.min(x)-5,np.max(x)+5])
    b = (w[1] * a +w[0])
    plt.plot(a, b, color="green")
    plt.scatter(x,y,s=0.5)
    plt.axis([np.min(x)-5, np.max(x)+5, np.min(y)-5, np.max(y)+5])
    plt.draw()
    plt.pause(0.1)
    fig.clear()
    gls=grad_loss(w)
    ls=loss(w)
plt.pause(3)
a = np.array([np.min(x)-5,np.max(x)+5])
plt.scatter(x,y,s=3)
b = (w[1] * a +w[0])
plt.plot(a, b, color="green")

plt.axis([np.min(x)-5, np.max(x)+5, np.min(y)-5, np.max(y)+5])
plt.show()


