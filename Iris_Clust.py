#final code to be in git hub for iris data set

import  random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
###################################################################
def random_centroids(points, K,r):
    ctds = []
    #Place K centroids at random locations
    for i in range(K):
        m=len(points)-1
        random.seed(r+i)
        p=random.randint(0, m)

        cent=points[p]
        ctds.append(cent)
    return ctds
###################################################################
def assign_cluster(points, ctds):
    Memberships = []

    for dp in points:
      dis_from_clus = []

      for c in ctds:
          d=np.linalg.norm(np.array(dp) - np.array(c))
          dis_from_clus.append(d)

      p_mem= np.argmin(dis_from_clus)
      Memberships.append((p_mem))

    return Memberships
#########################################################
def new_centroid(ctds,points,Memberships):
    new_ctds = []
    ps=[]
    Memberships=np.array(Memberships)

    for c in range(len(ctds)):

        c_points=[index for index, value in enumerate(Memberships) if value == c]

        for i in (c_points):
            ps.append(points[i])
        new_ctds.append(np.mean(ps,axis=0))

    return  new_ctds

########################################################
def Loss(new_ctds,points,Memberships):
    J=0.0
    #print("Memberships ")
    #print(Memberships)
    for p in range(len(points)):
        k=Memberships[p]
        mu= new_ctds[k]
        d=(np.linalg.norm(np.array(p) - np.array(mu)))**2
        J=J+d
    return J

#############################################
def fit(point,k,iter,eps,r):
    ct2=random_centroids(points,k,r)
    i = 1
    L=[]
    w=[]
    Mem = assign_cluster(point, ct2)
    ct2 = new_centroid(ct2, point, Mem)
    L.append(  Loss(ct2, point, Mem))
    print(L)
    e=1
    w.append(e)
    while(i<iter and e>= eps): #and e>= eps
        Mem = assign_cluster(point, ct2)
        ct2 = new_centroid(ct2, point, Mem)
        L.append(Loss(ct2, point, Mem))
        e = (abs(L[i]-L[i-1]))/ L[i-1]
        w.append(e)
        i = i+1
    return ct2,Mem,L
###############################################################

if __name__ == '__main__':

     points= np.loadtxt("iris.data",delimiter=",",usecols=[0,1,2,3])
     k=3
     iter=100
     eps=.00001
     h=6
     LL=[]
 #for k in [2,3,4,5]: # use this loop for elbow method

     points=np.array(points)
     c,m,L=fit(points,k,iter,eps,h)

     print("num of iter ")

     LL.append(min(L))
     pipeline2 = Pipeline([ ('pca', PCA(n_components=2))])
     points = pipeline2.fit_transform(points)
     points = pd.DataFrame(points)
     x = points[0]
     y = points[1]

     c = pipeline2.fit_transform(c)
     c = pd.DataFrame(c)
     centroids_x=c[0]
     centroids_y= c[1]

    # Single diagram
#################
     plt.plot(centroids_x, centroids_y, c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=18)
     plt.scatter(x, y, c=m)
     plt.xlabel("X1")
     plt.ylabel("X2")
     plt.show()
#########   Loss
     plt.xlabel("No. of iteration")
     plt.ylabel("Loss Function")
     plt.plot(np.arange(len(L)), L)
     plt.show()


    ###########End Sub plotting####################

     #plt.xlabel("K")

     #plt.ylabel("Loss Function")
     #df = pd.read_csv("iris.data", sep=",", header=None)
     #df.describe()
     #plt.show()
    #plt.title("K-Choice Visualization using Elbow Method")
    #plt.xlabel("K")
    #plt.ylabel("Loss Function")



