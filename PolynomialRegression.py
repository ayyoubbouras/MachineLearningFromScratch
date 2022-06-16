import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp 
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.optimize import minimize_scalar



x=list()
df=pd.read_csv(r'dataPolynomial.csv')
x=np.array(df['0']) 
y=np.array(df['1']) 
m=len(x)


def addDegre(X,i):
    x=np.array([1,X])
    j=2
        
    while(i>1):
        x= np.append(x, np.array([X**j]), axis=0)
        j=j+1
        i=i-1
    return x

def draw_Polynomial_Line(x,y,w,a,k):
    somme=np.zeros((50,))+w[0]
    i=1
    degre=k
    while k>0:
        somme=somme+w[i]*np.power(a,i)
        k=k-1
        i=i+1
    b=somme 
    ax.set_title("Polynome de degre %s" %degre,fontsize=14)   
    lines=ax.plot(a, b, color="green")
    plt.pause(0.0000001)
    lines=[]
    ax.lines.pop(0)   

def draw_Polynomial_Line_solution(x,y,w,a,k):
    somme=np.zeros((50,))+w[0]
    i=1
    degre=k
    while k>0:
        somme=somme+w[i]*np.power(a,i)
        k=k-1
        i=i+1
    b=somme 
    ax.set_title("Polynome de degre %s est La solution Finale" %degre,fontsize=14)   
    lines=ax.plot(a, b, color="green")
    plt.pause(6)
    lines=[]
    ax.lines.pop(0)  

def grad_loss(w,x,y,k):
    S=0
    for i,j in zip(x,y):
        S+=(np.dot(w,addDegre(i,k))-j)*addDegre(i,k)
    return (2*S)/m
def loss(w,x,y,k):
    S=0
    for i,j in zip(x,y):
        S+=(np.dot(w,addDegre(i,k))-j)**2
    return S/m

def alphaSearch(w,x,y,k):
    grad = grad_loss(w,x,y,k)
    a = sp.Symbol("a")
    w_k = w - a*grad
    f = str(loss(w_k,x,y,k))
    def fun(a):
        fu = eval(f)
        return fu
    alpha = minimize_scalar(fun)
    return alpha.x


figure, ax = plt.subplots(figsize=(10, 8))

# setting x-axis label and y-axis label
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.axis([np.min(x)-5, np.max(x)+5, np.min(y)-5, np.max(y)+5])


def Regression(w,x,y,k):
    
    grad=grad_loss(w,x,y,k)
    ls=loss(w,x,y,k)
    numIter=0
    a = np.linspace(np.min(x)-5, np.max(x)+5, 50)
    
    while numIter<38:
        alpha=alphaSearch(w,x,y,k)
        print(alpha , k)
        w=w-alpha*grad
        draw_Polynomial_Line(x,y,w,a,k)
        grad=grad_loss(w,x,y,k)
        numIter=numIter+1
    ls=loss(w,x,y,k)
    return  ls,w    
            



def PolynomialRegression(x,y):
    k=1
    w = np.zeros((k+1,))
    W=list()
    Ls_old,ww=Regression(w,x,y,k)
    W.append(ww)
    print(Ls_old)
    k=k+1
    
    while True:
        w = np.zeros((k+1,))
        Ls_new,ww=Regression(w,x,y,k)
        W.append(ww)
        print(Ls_new)
        if Ls_new<Ls_old:
            k=k+1
            Ls_old=Ls_new
            
            print("we continue at degre : ",k)
        else:
            print("we stopped at degre : ",k)
            a = np.linspace(np.min(x)-5, np.max(x)+5, 50)
            draw_Polynomial_Line_solution(x,y,W[k-2],a,k-1)
            break
global lines
lines=[]
ax.scatter(x,y,s=10)
PolynomialRegression(x,y) 
plt.show()       









