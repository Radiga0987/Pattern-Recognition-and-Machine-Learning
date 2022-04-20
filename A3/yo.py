import numpy as np
import os
import time

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
    cluster_nums = [0]*len(data)

    for n in range(len(data)):
        min_dist = float("inf")
        data_n = data[n]
        for K in range(k):
            if Euclidean_Distance(data_n,means[K]) < min_dist :
                min_dist = Euclidean_Distance(data_n,means[K])
                cluster_nums[n] = K+1

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

    for i in range(num_iterations):
        cluster_nums = Assign_Cluster(k,data,means)
        if i==0:
            # print(cluster_nums)
            pass
        if cluster_nums == old_cluster_nums:
            print(cluster_nums)
            print(old_cluster_nums)
            print(i)
            break
        means = Update_Mean(k,cluster_nums,data)
        old_cluster_nums = cluster_nums

t=time.time()
Find_Centroids(5,data_coast,10)
print(time.time()-t)
