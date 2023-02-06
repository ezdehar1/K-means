
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy

def hierarchicall(points):

    hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    hierarchical_cluster.fit_predict(points)
    points = pd.DataFrame(points)
    plt.scatter(points[0], points[1], c=hierarchical_cluster.labels_,)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def gmmc(points):
    pipeline2 = Pipeline([('pca', PCA(n_components=2))])
    points = pipeline2.fit_transform(points)
    pipeline = Pipeline([('scaling', StandardScaler())])
    points = pipeline.fit_transform(points)
    '''

    '''
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=50)

    gmm.fit(points)

    # Assign a label to each sample
    labels = gmm.predict(points)
    points = pd.DataFrame(points)
    # plt.plot(points[0], points[1], c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=18)
    plt.scatter(points[0], points[1], c=labels, )
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
def Spectclus(points):
    sp = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
    pipeline = Pipeline([('scaling', StandardScaler())])
    points = pipeline.fit_transform(points)
    sp.fit(points)
    labels = sp.labels_

    points = pd.DataFrame(points)
    # plt.plot(points[0], points[1], c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=18)
    plt.scatter(points[0], points[1], c=sp.labels_)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    points = numpy.loadtxt("OverlapGus", delimiter=",", usecols=[0, 1])

    hierarchicall(points)
    #Spectclus(points)
    #gmmc(points)
