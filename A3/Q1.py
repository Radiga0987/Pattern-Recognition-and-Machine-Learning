import numpy as np
import os
import time
import matplotlib.pyplot as plt
# dir_list = os.listdir('Features//coast//train')
# data_coast = []
# for file in dir_list:
#     data_coast.extend(np.loadtxt('Features/coast/train/'+file))
# data_coast = np.array(data_coast)

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
            # print(cluster_nums)
            # print(old_cluster_nums)
            # print(i)
            break
        means = Update_Mean(k,cluster_nums,data)
        old_cluster_nums = cluster_nums

    cov=[]
    pi_k=[]

    for j in range(k):
        indexes = np.where(np.array(cluster_nums)==j+1)
        cov.append(covariance(data[indexes],means[j],flag=0))
        pi_k.append(len(indexes[0])/len(data)) 
    return np.array(means),np.array(cov),np.array(pi_k)

def covariance(x,u,flag=1,r_k=[],N_k=[]):
    if flag==0:
        r_k = [1]*len(x)
        N_k = len(x)-1
    out = np.zeros((len(x[0]),len(x[0])))
    for i,v in enumerate(x):
        vec = (v-u)
        vec = vec.reshape((len(vec),1))
        out += r_k[i] * np.dot(vec,vec.T)
    return (1/N_k) * out


from scipy.stats import multivariate_normal
def Normal_distribution(x,u,E):     #TO DO : Make this more efficient
    lol = multivariate_normal.pdf(x,u,E,allow_singular=True)
    return lol
    E_inv = np.linalg.pinv(E)
    x_m = x-u
    x_m = x_m.reshape((len(x_m),1))
    exp_arg = -1/2 * np.dot(np.dot(x_m.T,E_inv),x_m)[0][0]
    cond = np.linalg.cond(E)
    # v1 = pow(2*np.pi,len(x)/2)
    # v2 = np.sqrt(np.linalg.det(E))
    # v3 = np.exp(exp_arg)
    gaus = 1/pow(2*np.pi,len(x)/2)/np.sqrt(abs(np.linalg.det(E))) * np.exp(exp_arg)
    return gaus

def gmm(data,k,pi_k,mean,cov,num_iterations=10):
    for ite in range(num_iterations):
        #Expectation
        r_nk = np.zeros((len(data),k))
        for i,x in enumerate(data):
            numerators = np.array([pi_k[j] * Normal_distribution(x,mean[j],cov[j]) for j in range(k)])
            denominators = sum(numerators)
            r_nk[i,:] = numerators/denominators

        #Maximisation
        new_mean = mean.copy()
        N = len(data)
        for j in range(k):
            N_k = sum(r_nk[:,j])
            pi_k[j] = N_k/N
            new_mean[j] = sum((data.T*r_nk[:,j]).T)/N_k
            # new_mean[k] = 1/N_k*np.dot(r_nk[:,j].T,data)
            cov[j] = covariance(data,mean[j],1,r_nk[:,j],N_k)
        mean = new_mean
    return mean,cov,pi_k

def find_probabiltity(x,mean_k,cov_k,pi_k,k):
    prob = 0
    for feature_vector in x:
        val = [pi_k[j] * Normal_distribution(feature_vector,mean_k[j],cov_k[j]) for j in range(k)]
        prob+=np.log10(sum(val))
    return prob
    

def classification(x,means,covs,pi,k):
    classify = -1
    for i in range(len(means)):
        P_x = find_probabiltity(x,means[i],covs[i],pi[i],k)
        if i==0 or P_x > P_max:
            P_max = P_x
            classify = i
    return classify

# def classification_partB(x,means,covs,pi,k):
#     p1 = find_probabiltity([x],)
#     return classify



# mean,cov,pi_k = Find_Centroids(k,data_coast,10)
# # print(means,pi_k)
# print(pi_k)
# new_mean,new_cov,new_pi_k=gmm(data_coast,k,pi_k,mean,cov,2)
# # print(means)
# means.append(new_mean)
# covs.append(new_cov)
# pi.append(new_pi_k)
# print("trained")

def partA():
    start = time.time()
    classes = ['coast','forest','highway','mountain','opencountry']
    train = []
    dev = []

    for clas in classes:
        lst = []
        dir_list = os.listdir('Features//'+clas+'//train')
        for file in dir_list:
            lst.extend(np.loadtxt('Features/'+clas+'/train/' + file))
        train.append(np.array(lst))

        lst = []
        dir_list = os.listdir('Features//'+clas+'//dev')
        for file in dir_list:
            lst.append(np.loadtxt('Features/'+clas+'/dev/' + file))
        dev.append(np.array(lst))

    means,covs,pi = [],[],[]
    k= 20
    km_num_iterations = 15
    gmm_num_iterations = 4

    for clas_data in train:
        mean,cov,pi_k = Find_Centroids(k,clas_data,km_num_iterations)
        new_mean,new_cov,new_pi_k=gmm(clas_data,k,pi_k,mean,cov,gmm_num_iterations)
        means.append(new_mean)
        covs.append(new_cov)
        pi.append(new_pi_k)
        print("trained")

    count_correct = 0
    count_dev_files = 0
    for i,test_clas in enumerate(dev):
        print("class",i)
        for test_case in test_clas:
            classify = classification(test_case,means,covs,pi,k)
            print(classify)
            count_dev_files +=1
            if classify == i:
                count_correct += 1
        print()
    print("Accuracy = ", count_correct/count_dev_files)
    print(time.time()-start)

def partB():
    start = time.time()
    
    with open("Synthetic_Dataset/train.txt") as f:
        train_data = [[float(val) for val in line.strip().split(',')] for line in f]
    train_data = np.array(train_data)
    with open("Synthetic_Dataset/dev.txt") as f:
        dev_data = [[float(val) for val in line.strip().split(',')] for line in f]
    dev_data = np.array(dev_data)

    train = []
    class1,class2=[],[]
    for i in range(len(train_data)):
        if train_data[i][2]==1:
            class1.append(train_data[i][:2])
        if train_data[i][2]==2:
            class2.append(train_data[i][:2])
    class1=np.array(class1)
    class2=np.array(class2)
    train.append(class1)
    train.append(class2)

    
    # plt.plot(class1[:,0],class1[:,1],'.')
    # plt.plot(class2[:,0],class2[:,1],'.')

    dev = []
    class1,class2=[],[]
    for i in range(len(dev_data)):
        if dev_data[i][2]==1:
            class1.append(dev_data[i][:2])
        if dev_data[i][2]==2:
            class2.append(dev_data[i][:2])
    class1=np.array(class1)
    class2=np.array(class2)
    dev.append(class1)
    dev.append(class2)

    # plt.plot(class1[:,0],class1[:,1],'.')
    # plt.plot(class2[:,0],class2[:,1],'.')
    # plt.show()

    means,covs,pi = [],[],[]
    k=20
    km_num_iterations = 10
    gmm_num_iterations = 3

    for clas_data in train:
        mean,cov,pi_k = Find_Centroids(k,clas_data,km_num_iterations)
        new_mean,new_cov,new_pi_k=gmm(clas_data,k,pi_k,mean,cov,gmm_num_iterations)
        # plt.plot(class1[:,0],class1[:,1],'.')
        # plt.plot(class2[:,0],class2[:,1],'.')
        # plt.plot(new_mean[:,0],new_mean[:,1],'.')
        # plt.show()
        means.append(new_mean)
        covs.append(new_cov)
        pi.append(new_pi_k)
        print("trained")

    count_correct = 0
    count_dev_files = 0
    for i,test_clas in enumerate(dev):
        print("class",i)
        for test_case in test_clas:
            classify = classification([test_case],means,covs,pi,k)
            # print(classify)
            count_dev_files +=1
            if classify == i:
                count_correct += 1
        print()
    print("Accuracy = ", count_correct/count_dev_files)
    print(time.time()-start)


partA()
# partB()
