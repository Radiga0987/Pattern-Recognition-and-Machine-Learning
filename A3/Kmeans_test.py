import numpy as np
import os
import time

from sklearn import cluster

dir_list = os.listdir('Features//coast//train')
data_coast = []
for file in dir_list:
    data_coast.extend(np.loadtxt('Features/coast/train/'+file))
data_coast = np.array(data_coast)


def Euclidean_Distance(v1,v2):
    d = np.sqrt(sum((v1-v2)**2))
    return(d)

def Random_Means_Initialization(k,data):
    means = []
    for i in np.random.randint(0, len(data),k):
        means.append(data[i])

    return means

def Assign_Cluster(k,data,means):
    means = np.array(means)

    distances = np.array([]).reshape(data.shape[0],0)

    # finding distances of feature vecs from each mean
    for K in range(k):
        dist = np.sum((data-means[K,:])**2,axis=1)
        distances = np.c_[distances,dist]

    # Alloting each point to nearest mean
    cluster_nums = np.argmin(distances,axis=1) + 1

    return cluster_nums

def Update_Mean(k,cluster_nums,data):  #TO DO : Make this more efficient 
    means = [[0 for j in range(len(data[0]))] for i in range(k)]
    count = [0]*k
    for n in range(len(data)):
        means[cluster_nums[n]-1] += data[n]
        count[cluster_nums[n]-1] += 1

    return [[x/count[i] for x in means[i]] if count[i]!=0 else [0]*len(data[0]) for i in range(k)]

def Find_Centroids(k,data,num_iterations):
    means = Random_Means_Initialization(k,data)
    old_cluster_nums = [0]*len(data)
    np_data = np.array(data)

    for i in range(num_iterations):
        cluster_nums = Assign_Cluster(k,np_data,means)
        if np.sum(cluster_nums - old_cluster_nums) == 0:
            break
        means = Update_Mean(k,cluster_nums,data)
        old_cluster_nums = cluster_nums

    return means,cluster_nums

means,cluster_nums = Find_Centroids(5,data_coast,10)

print(len(means))
print(cluster_nums)