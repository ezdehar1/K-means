import  random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
import  argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#### This function is used to assign random centroids
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

### this function assign ech point to a to the nearest cluster
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

#### Calculate new centriods
def new_centroid(ctds,points,Memberships):
    new_ctds = []
    ps=[]
    Memberships=np.array(Memberships)

    for c in range(len(ctds)):

        c_points=[index for index, value in enumerate(Memberships) if value == c]
        #print(c_points)
        for i in (c_points):
            ps.append(points[i])
        new_ctds.append(np.mean(ps,axis=0))

    return  new_ctds

###### Calcualte Loss Function
def Loss(new_ctds,points,Memberships):
    J=0.0

    for p in range(len(points)):
        k=Memberships[p]
        mu= new_ctds[k]
        d=(np.linalg.norm(np.array(p) - np.array(mu)))**2
        J=J+d
    return J



#### Main body of Kmeans
def fit(point,k,iter,eps,r):
    ct2=random_centroids(point,k,r)
    i = e= 1
    L=[]
    w=[]
    Mem = assign_cluster(point, ct2)
    ct2 = new_centroid(ct2, point, Mem)
    L.append(  Loss(ct2, point, Mem))

    w.append(e)
    while(i<iter and e>= eps): #and e>= eps
        Mem = assign_cluster(point, ct2)
        ct2 = new_centroid(ct2, point, Mem)
        L.append(Loss(ct2, point, Mem))
        e = (abs(L[i]-L[i-1]))/ L[i-1]
        w.append(e)
        i = i+1
    return ct2,Mem,L

def main(args):


    DS_Path=args.DS_Path
    k= args.K
    iter = args.iter
    eps = args.eps
    h = args.seed
    LL = []
    points = np.loadtxt(DS_Path, delimiter=",", usecols=[0, 1])
    pipeline = Pipeline([('scaling', StandardScaler())])
    points = pipeline.fit_transform(points)

    points = np.array(points)
    c, m, L = fit(points, k, iter, eps, h)

    print("num of iter ")
    print((L))
    print(len(L))
    # print(points)
    LL.append(min(L))

    points = pd.DataFrame(points)
    x = points[0]
    y = points[1]
    c = pd.DataFrame(c)
    centroids_x = c[0]
    centroids_y = c[1]

    fig, axs = plt.subplots(2)
    axs[0].scatter(x, y, c=m)
    axs[0].plot(centroids_x, centroids_y, c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=22)
    axs[1].plot(np.arange(len(L)), L)

    axs[0].set_xlabel("PCA for Iris DataSet")
    axs[1].set_xlabel("Iteration No")
    axs[1].set_ylabel("Loss Fucntion")
    plt.show()


##################StandBy code for single plot
#################
# plt.plot(centroids_x, centroids_y, c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=18)
# plt.scatter(x, y, c=m)
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()
#########   Loss
# plt.xlabel("No. of iteration")
# plt.ylabel("Loss Function")
# plt.plot(np.arange(len(L)), L)
# plt.show()
###########

def parse_args():
    ## Fee free to change the values of the defualt parameters to run other experiments
    parser=argparse.ArgumentParser()
    parser.add_argument("--K", type=int,default=3,help="Number of clusters")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=5, help="Seed to intilize cluster")
    parser.add_argument("--eps", type=float, default=.00001, help="thershold for loss fucntion")
    parser.add_argument("--DS_Path", type=str, default="Guss",help="Path of the data set")

    args=parser.parse_args()
    return args


# Pleasae Use this code for Guss,OverlapGus, moons, circles datasets Only (for iris data set use iris_clust file)

if __name__ == '__main__':

     main(parse_args())


