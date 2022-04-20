from operator import index
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
from scipy.stats import multivariate_normal as norm_gaus
# Normal_distribution = norm_gaus.pdf
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
    for i in range(k):
        indexes = np.where(np.array(cluster_nums)==i+1)
        if len(indexes[0]) == 0:
            means[i] = data[np.random.randint(0,len(data))]

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
    

def intialization(data,means,cluster_nums,k,flag_diag=0):
    cov=[]
    pi_k=[]
    mean = []
    K = k
    for j in range(k):
        indexes = np.where(np.array(cluster_nums)==j+1)
        if len(indexes[0]) ==0 :
            K=K-1
            continue
        mean.append(means[j])
        cov.append(covariance(data[indexes],means[j],1,[],[],flag_diag))
        pi_k.append(len(indexes[0])/len(data)) 
    return np.array(mean),np.array(cov),np.array(pi_k),K

def plot_Distortion_K(max_K,data):
    distortion = []
    for k in range(1,max_K+1):
        distortion.append(0)
        means,cluster_nums = Find_Centroids(k,data,30)
        for n in range(len(data)):
            distortion[-1] += Euclidean_Distance(data[n],means[cluster_nums[n]-1])
    plt.plot(np.linspace(1,max_K,max_K),distortion)
    plt.show()


def covariance(x,u,flag_initialise=0,r_k=[],N_k=[],flag_diag=0):
    if flag_initialise==1:
        r_k = [1]*len(x)
        N_k = len(x)-1
    out = np.zeros((len(x[0]),len(x[0])))
    for i,v in enumerate(x):
        vec = (v-u)
        vec = vec.reshape((len(vec),1))
        out += r_k[i] * np.dot(vec,vec.T)
    covariance_matrix = (1/N_k) * out
    if flag_diag == 1:
        covariance_matrix = np.diag(np.diag(covariance_matrix))
    return covariance_matrix

def Normal_distribution(x,u,E):     
    dim = len(E)
    x_m = x-u
    E_inv = np.linalg.pinv(E)
    temp1 = np.dot(x_m,E_inv)
    temp2 = temp1 * x_m
    exp_arg = -1/2 * sum(temp2.T)
    gaus = 1/pow(2*np.pi,dim/2)/np.sqrt(abs(np.linalg.det(E))) * np.exp(exp_arg)
    return gaus

def gmm(data,k,pi_k,mean,cov,num_iterations=10,flag_diag=0):
    for ite in range(num_iterations):
        #Expectation
        r_nk = np.zeros((len(data),k))
        numerators = np.zeros((len(data),k))
        for j in range(k):
            numerators[:,j] = Normal_distribution(data,mean[j],cov[j])
        numerators = numerators*pi_k
        denominators = sum(numerators.T)
        r_nk = (numerators.T/denominators).T

        #Maximisation
        new_mean = mean.copy()
        N = len(data)
        for j in range(k):
            N_k = sum(r_nk[:,j])
            pi_k[j] = N_k/N
            new_mean[j] = sum((data.T*r_nk[:,j]).T)/N_k
            # new_mean[k] = 1/N_k*np.dot(r_nk[:,j].T,data)
            cov[j] = covariance(data,mean[j],0,r_nk[:,j],N_k,flag_diag)
        mean = new_mean
    return mean,cov,pi_k

# def find_probabiltity(x,mean_k,cov_k,pi_k,k):
#     prob = 0
#     for feature_vector in x:
#         val = [pi_k[j] * Normal_distribution(feature_vector,mean_k[j],cov_k[j]) for j in range(k)]
#         prob+=np.log10(sum(val))
#     return prob
    

# def classification(x,means,covs,pi,k):
#     classify = -1
#     P_max = 0
#     for i in range(len(means)):
#         P_x = find_probabiltity(x,means[i],covs[i],pi[i],k)
#         if i==0 or P_x > P_max:
#             P_max = P_x
#             classify = i
#     return classify

def find_prob(no_of_classes,test_data,means,covs,pi,K_s):
    P = np.zeros((len(test_data),no_of_classes))
    for cls in range(no_of_classes):
            numerators = np.zeros((len(test_data),K_s[cls]))
            for j in range(K_s[cls]):
                numerators[:,j] = Normal_distribution(test_data,means[cls][j],covs[cls][j])
            numerators = numerators*pi[cls]
            P[:,cls] = sum(numerators.T)
    print("lol")
    return P


def classification2(P,priors):
    P_max = 0
    Probs = []
    classify = -1
    for i in range(len(P[0])):
        prob=sum([np.log10(val) for val in P[:,i]])
        prob += np.log10(priors[i])
        Probs.append(prob)
        if i==0 or prob > P_max:
            P_max = prob
            classify = i
    return classify,Probs

def confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title("Title", fontsize=12)
    plt.show()

def ROC_DET(S_list,class_labels):
    Scores_list = []
    for case_no in range(len(S_list)):  #For Loop for all cases
        S = S_list[case_no]
        S = np.array(S)
        Scores = sorted(S.flatten())   #Scores are sorted for thresholding
        Scores_list.append(S.T.flatten())
        
        TPR = [0]*len(Scores)   
        FPR = [0]*len(Scores) 
        count=0
        for threshold in Scores:
            TP,FP,TN,FN = 0,0,0,0
            for i in range(len(S)):
                for j in range(len(S[0])):
                    if S[i][j] >= threshold:        #Classifying As Positive
                        if class_labels[i] == j+1: TP+=1
                        else:   FP+=1
                    else:
                        if class_labels[i] == j+1: FN+=1
                        else:   TN+=1
            TPR[count] = TP/(TP+FN)     #True Positive Rate
            FPR[count] = FP/(FP+TN)     #False Positive Rate

            count+=1
        plt.plot(FPR,TPR,label="Case "+str(case_no))
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic curve")
    plt.show()

    y_true = [0]*(len(S_list[0][0])*len(class_labels))
    count = 0
    for i in range(len(S[0])):
        for j in range(len(class_labels)):
            if class_labels[j] == i+1:
                y_true[count]=1
            count += 1
    plt.figure(figsize=(8,5))    
    for i in range(len(S_list)):
        S = Scores_list[i]
        y_true = np.array(y_true)
        fpr, fnr, thresholds = det_curve(y_true, S)
        plt.plot(fpr,fnr,label="Case "+str(i+1))
        plt.yscale('logit')
        plt.xscale('logit')
    plt.legend()
    plt.xlabel("False Postive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Detection Error Tradeoff curve")
    plt.show()

def contour(means,covs,k):
    x1 = np.linspace(-18,18,100)
    x2 = np.linspace(-18,18,100)
    X,Y = np.meshgrid(x1,x2)
    Space = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            Space.append([X[i][j],Y[i][j]])
    Space = np.array(Space)
    Z1 = np.zeros((len(X),len(X[0])))
    plt.figure(figsize=(7,7))
    ax = plt.subplot()
    for i in range(k):
        Pdf = Normal_distribution(Space,means[i],covs[i])
        # indexes = np.where(Pdf<0.1)
        # Pdf[indexes] = 0
        for i in range(len(X)):
            Z1[i,:] = Pdf[i*len(X):(i+1)*len(X)]
        ax.contour(X,Y,Z1)
    plt.show()

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

def partA(k= 30, km_num_iterations = 10, gmm_num_iterations = 4, flag_diag = 0):
    start = time.time()
    classes = ['coast','forest','highway','mountain','opencountry']
    train = []
    dev = []
    test = []
    mean_train_data = []
    denoms = []
    priors = np.zeros(len(classes))

    for i,cls in enumerate(classes):
        lst = []
        dir_list = os.listdir('Features//'+cls+'//train')
        priors[i] = len(dir_list)
        for file in dir_list:
            lst.extend(np.loadtxt('Features/'+cls+'/train/' + file))
        lst = np.array(lst)
        train.append(lst)

        lst = []
        lst2 = []
        dir_list = os.listdir('Features//'+cls+'//dev')
        for file in dir_list:
            lst.append(np.loadtxt('Features/'+cls+'/dev/' + file))
            lst2.extend(np.loadtxt('Features/'+cls+'/dev/' + file))
        dev.append(np.array(lst))
        test.append(np.array(lst2))

    # plot_Distortion_K(50,train[0])

    priors = priors/sum(priors)
    means,covs,pi,K_s = [],[],[],[]

    #Training
    for i,clas_data in enumerate(train):
        mean,cluster_nums = Find_Centroids(k,clas_data,km_num_iterations)
        mean,cov,pi_k,K = intialization(clas_data,mean,cluster_nums,k,flag_diag)
        print("k_means")
        new_mean,new_cov,new_pi_k=gmm(clas_data,K,pi_k,mean,cov,gmm_num_iterations,flag_diag)
        means.append(new_mean)
        covs.append(new_cov)
        pi.append(new_pi_k)
        K_s.append(K)
        print("trained")

    #Test_Evaluation
    count_correct = 0
    count_dev_files = 0
    S = []
    class_labels = []
    conf_matrix = np.zeros((len(train),len(train)))
    for i,test_clas in enumerate(dev):
        _S = [[] for j in range(len(test_clas))]
        class_labels.extend([i+1]*len(test_clas))
        print("class",i)
        data = test[i]
        P = find_prob(len(train),data,means,covs,pi,K_s)

        for j in range(len(test_clas)):
            # classify = classification(test_case,means,covs,pi,k)
            classify2,Probs = classification2(P[j*36:(j+1)*36,:],priors)
            _S[j] = Probs
            # if classify2!=classify:
                # print("pain")
            # print(classify2)
            # print(classify)
            count_dev_files +=1
            if classify2 == i:
                count_correct += 1
            conf_matrix[i][classify2] += 1
        S.extend(_S)
        print()
    print("Accuracy = ", count_correct/count_dev_files*100,"%")
    confusion_matrix(conf_matrix)
    print(time.time()-start)
    return S,class_labels

def partB(k= 15, km_num_iterations = 10, gmm_num_iterations = 4, flag_diag=0):
    start = time.time()
    
    with open("Synthetic_Dataset/train.txt") as f:
        train_data = [[float(val) for val in line.strip().split(',')] for line in f]
    train_data = np.array(train_data)
    with open("Synthetic_Dataset/dev.txt") as f:
        dev_data = [[float(val) for val in line.strip().split(',')] for line in f]
    dev_data = np.array(dev_data)

    #Separating Train data into respective classes
    train = []
    class1,class2=[],[]
    for i in range(len(train_data)):
        if train_data[i][2]==1:
            class1.append(train_data[i][:2])
        if train_data[i][2]==2:
            class2.append(train_data[i][:2])
    priors = np.zeros(2)
    priors[0] = len(class1)
    priors[1] = len(class2)
    priors = priors/sum(priors)
    class1=np.array(class1)
    class2=np.array(class2)
    train.append(class1)
    train.append(class2)

    
    # plt.plot(class1[:,0],class1[:,1],'.')
    # plt.plot(class2[:,0],class2[:,1],'.')

    #Separating Test data into respective classes
    dev = []
    dev_class1,dev_class2=[],[]
    for i in range(len(dev_data)):
        if dev_data[i][2]==1:
            dev_class1.append(dev_data[i][:2])
        if dev_data[i][2]==2:
            dev_class2.append(dev_data[i][:2])
    dev_class1=np.array(dev_class1)
    dev_class2=np.array(dev_class2)
    dev.append(dev_class1)
    dev.append(dev_class2)

    # plt.plot(class1[:,0],class1[:,1],'.')
    # plt.plot(class2[:,0],class2[:,1],'.')
    plt.show()

    means,covs,pi,K_s = [],[],[],[]
    # plot_Distortion_K(50,train[0])
    #Training
    for clas_data in train:
        mean,cluster_nums = Find_Centroids(k,clas_data,km_num_iterations)
        mean,cov,pi_k,K = intialization(clas_data,mean,cluster_nums,k,flag_diag)
        new_mean,new_cov,new_pi_k=gmm(clas_data,K,pi_k,mean,cov,gmm_num_iterations,flag_diag)
        # plt.plot(class1[:,0],class1[:,1],'.')
        # plt.plot(class2[:,0],class2[:,1],'.')
        # plt.plot(new_mean[:,0],new_mean[:,1],'.')
        # plt.show()
        K_s.append(K)
        means.append(new_mean)
        covs.append(new_cov)
        pi.append(new_pi_k)
        print("trained")

    contour(means[0],covs[0],K_s[0])
    contour(means[1],covs[1],K_s[1])

    #Test Evaluation
    count_correct = 0
    count_dev_files = 0
    S = []
    class_labels = []
    conf_matrix = np.zeros((len(train),len(train)))
    for i,test_clas in enumerate(dev):
        print("class",i)
        P = find_prob(len(train),test_clas,means,covs,pi,K_s)
        S.extend(list(P))
        class_labels.extend([i+1]*len(test_clas))

        for j in range(len(test_clas)):
            # classify = classification([test_case],means,covs,pi,k)
            classify2,Probs = classification2(P[j:(j+1),:],priors)
            # if classify2!=classify:
            #     print("pain")
            # print(classify2)
            # print(classify)
            count_dev_files +=1
            if classify2 == i:
                count_correct += 1
            conf_matrix[i][classify2] += 1
        print()
    confusion_matrix(conf_matrix)
    print("Accuracy = ", count_correct/count_dev_files*100,"%")
    print(time.time()-start)
    return(S,class_labels)

S_list = []
S_,class_labels_b = partA(k=30, km_num_iterations = 25, gmm_num_iterations = 4,flag_diag=0)
S_list.append(S_)
S_,class_labels_b = partA(k=30, km_num_iterations = 25, gmm_num_iterations = 4,flag_diag=1)
S_list.append(S_)
ROC_DET(S_list,class_labels_b)

S_list = []
# S_,class_labels_b = partB(k=20, km_num_iterations = 20, gmm_num_iterations = 30, flag_diag=0)
# S_list.append(S_)
# S_,class_labels_b = partB(k=25, km_num_iterations = 10, gmm_num_iterations = 4, flag_diag=1)
# S_list.append(S_)
# ROC_DET(S_list,class_labels_b)