import numpy as np
from sklearn.decomposition import PCA


def covariance(x,u,flag_initialise=0,r_k=[],N_k=[],flag_diag=0):
    if flag_initialise==1:  #Computes the general form of Covariance Matrices
        r_k = [1]*len(x)
        N_k = len(x)-1
    out = np.zeros((len(x[0]),len(x[0])))
    for i,v in enumerate(x):
        vec = (v-u)
        vec = vec.reshape((len(vec),1))
        out += r_k[i] * np.dot(vec,vec.T)
    covariance_matrix = (1/N_k) * out
    
    #Returns a Diagonal Matrix
    if flag_diag == 1:
        covariance_matrix = np.diag(np.diag(covariance_matrix))
    return covariance_matrix


def pca_(data,L):
    cov = covariance(data,np.mean(data,axis=0),1,[],[],0)
    e,V = np.linalg.eig(cov)        #Finding Eigen values and vectors of martix A
    mag,e,V = map(np.array, zip(*sorted(zip(abs(e),e,V.T),reverse=True)))   #Sorting eigen vals and vecs
    V=V.T
    # for i in range(len(cov)):
    #     exp_var = np.sum(e[:i+1]) / np.sum(e)
    #     if exp_var > L:
    #         break
    # return V[:,:i]
    return V[:,:L]

def pca_transform(data,Q):
    data_transform = np.dot(Q.T,data.T)
    return data_transform.T

data = np.array([[1,2,3,3,4,2],[2,4,5,2,3,4],[2,1,3,4,1,4],[6,4,5,6,7,4]])
means = np.mean(data,axis=0)
maxs=np.max(data,axis=0)
mins = np.min(data,axis =0)
data = (data-means)/(maxs-means)

Q = pca_(data,5)
print(pca_transform(data,Q))

