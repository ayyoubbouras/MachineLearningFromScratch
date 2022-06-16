import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rcParams['contour.negative_linestyle'] = 'solid'
figure, ax = plt.subplots(figsize=(10, 10))

n=100
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel1(X, x, p=2):
    somme=list()
    for i in range(X.shape[0]):
        somme.append((1 + np.dot(X[i], x[i])) ** p)
    somme=np.array(somme)
    return somme

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p
X=list()
df=pd.read_csv(r'dataCircle.csv')
for i,j in zip(np.array(df['0']),np.array(df['1'])) :
	X.append([i,j])
X=np.array(X)
y=np.array(df['2']) 


for i in range(n):
    if y[i]==0:
        y[i]=-1



K_P = np.zeros((n, n))
K_L= np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K_P[i,j] = polynomial_kernel(X[i], X[j],2)

print( K_P.shape) 

for i in range(n):
    for j in range(n):
        K_L[i,j] = linear_kernel(X[i], X[j])

#print("Le kernel polynomial du degre 2 :", K_P)
#print("Le kernel lineaire :", K_L)



def f(x,w,b):
    if w[1]==0:
        w5=0.001
        return (-b-w[0]*x**2)/w5
    return (-b-w[0]*x**2)/w[1]

def kernelsum(alpha,y,i):
    somme=0
    for j in range(n):
        somme=somme+alpha[j]*y[j]*K_P[i,j]
    return somme  

def w_par_Kernel_Polynomial(alpha,y,X):
    somme=np.array([0,0])
    for j in range(n):
        somme=somme+alpha[j]*y[j]*X[j]
    return somme  
        
alpha=np.zeros((n,))
print("alpha avant les updates ",alpha)   
m=20
b=0
j=0
####################PERCEPTRON AVEC L'INTEGRATION DU KERNEL###########################################
while(j<=m):
    for i in range(n):
        if y[i]*(kernelsum(alpha,y,i)+b)<=0:
            alpha[i]=alpha[i]+1
            b=b+y[i]   
    j=j+1  
#######################################################################################################
print("alpha apres tous les updates ",alpha)    
w=w_par_Kernel_Polynomial(alpha,y,X)   
print("Le w resultat par la methode du kernel est :", w)


def kernelDecision(X,x,alpha):
    somme=0
    for i in range(n):
        somme=somme+alpha[i]*y[i]*polynomial_kernel(X[i], x)
    print(somme)
    if somme<0:
        classe=1
    else:
        classe=2
    return somme,classe  
        
       
'''         
ff= lambda x,y: kernelDecision(X,x,y,alpha) 
x=np.linspace(-2,2,10)
yy=np.linspace(-2,2,10)
x,yy=np.meshgrid(x,yy)
F= ff(x,yy)
ax.contour(x,yy,F)
print("XI :" ,xi.shape)
'''


n_x = 0.25
n_y = 0.25
x1 = np.arange(-2.0, 2.0, n_x)
x2 = np.arange(-2.0, 2.0, n_y)
xx1, xx2 = np.meshgrid(x1, x2)

f = np.zeros(xx1.shape)
for i in range(xx1.shape[0]):
    for j in range(xx1.shape[1]):
        #print(np.tile(np.array([xx1[i,j],xx2[i,j]]),(X.shape[0],1)))
        f[i,j] = (
                    (
                        alpha              
                        *y
                        *polynomial_kernel1(X, np.tile(np.array([xx1[i,j],xx2[i,j]]),(X.shape[0],1)),2) 
                    ).sum()
                 ) 

cs = ax.contour(x1, x2, f, 0, linewidths=4,colors='k')
ax.clabel(cs, inline=1, fontsize=15, fmt='%1.1f', manual=[(1,1)])


class1 = ( y== -1)
class2 = ( y== 1)

for i in np.arange(-1,1,0.1):
    ax.scatter(X[class1, 0] ,X[class1, 1],color='blue',label ='classe 1')
    ax.scatter(X[class2, 0] ,X[class2, 1],color='red',label ='classe 2')
    j=np.random.randint(-1, 1)/np.random.randint(1, 20)
    x=np.array([i,j])
    r,classe=kernelDecision(X,x,alpha)
    ax.set_title("Le point x appartient a classe %s" %(classe))
    ax.scatter(x[0],x[1],color='black',label ='point black')
    plt.legend()
    plt.pause(3.5)
    ax.cla()

plt.axis('scaled')
plt.show()

   





