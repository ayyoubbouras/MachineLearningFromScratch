
############################################################IMPORTATION NECCESSAIRE#######################################################################
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#########################################################################################################################################################






###############################Importatation du fichier 'binary.csv' et le transformer en numpy array##################################
X=list()
df=pd.read_csv(r'binary.csv')
for i,j in zip(np.array(df['gre']),np.array(df['gpa'])) :
	X.append([1,i,j])
X=np.array(X)
y=np.array(df['admit'])
#######################################################################################################################################



##### Feature Scaling la standarisation##########
sc = StandardScaler()
X = sc.fit_transform(X)
##################################################





#################loss function########################
def lossfunction(x, y, w):
    m = len(x)
    som =0
    for i in range(m):
         som += np.log(1 + np.exp(-y[i]*(np.dot(w, x[i]))))
        
    return som/m
#####################################################

    
#####gradient de loss function#########
def gradient(x, y, w):
    som = 0
    m = len(x)
    for i in range(m):
        t1 = (-y[i] * np.exp(-y[i] * np.dot(w, x[i])))
        t2 = 1/(1 + np.exp(-y[i] * np.dot(w, x[i])))
        
        som +=  (t1/t2) * x[i]
    return som/m
    
########sigmoid activation function#####################
def sigmoid(w,x,y):
    return 1/(1+np.exp(-x*w[1]-y*w[2]-w[0]))

########Regression Logistique########################################
def regressionLogistic(x, y, w, learning_rate= 0.4, tolerance = 0.001):
    cpt = 0
    gradloss = gradient(x,y, w)
    while np.linalg.norm(gradloss) > tolerance :
        w = w - learning_rate * gradloss
        #print(w)
        gradloss = gradient( x, y, w)
        cpt += 1
        print(np.linalg.norm(gradloss))
        
    loss = lossfunction(x, y, w)
    return w, loss, cpt
######################################################################  
    
#############################regressionLogistic() #########
#initialisation de w
w = [0,0,0]
W, loss, compteur = regressionLogistic(X, y, w) 
#############################################################


##################Visualisation##############################
ax.set_title("Regression Logistique", fontsize=15)
ax.scatter3D(X[:,1],X[:,2],y,c=y,s=25,marker="o")
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x,y=np.meshgrid(x,y)
ss=sigmoid(W,x,y)
ax.plot_surface(x,y,ss,alpha=0.4)
plt.show()
#############################################################




