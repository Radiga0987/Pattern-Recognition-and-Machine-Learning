import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
from sklearn.decomposition import PCA
# Normal_distribution = norm_gaus.pdf


def Euclidean_Distance(v1,v2):
    d = np.sqrt(sum((v1-v2)**2))
    return(d)

def Random_Means_Initialization(k,data):
    means = []
    # maxs = np.max(data,axis=0)
    # mins = np.min(data,axis=0)

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
    distortion = []

    for i in range(num_iterations):
        cluster_nums = Assign_Cluster(k,np_data,means)
        if np.sum(cluster_nums - old_cluster_nums) == 0:
            break
        means = Update_Mean(k,cluster_nums,data)
        distances = 0
        for n in range(len(data)):
            distances+=(Euclidean_Distance(data[n],means[cluster_nums[n]-1]))
        distortion.append(distances)
        old_cluster_nums = cluster_nums
    return means,cluster_nums,distortion
    
#Initialising the Means, Covariance Matrices and Mixture Weights
def intialization(data,means,cluster_nums,k,flag_diag=0):
    cov=[]
    pi_k=[]
    mean = []
    K = k
    #Using the data points that belong to a specefic cluster, the covariance and weights are computed
    for j in range(k):
        indexes = np.where(np.array(cluster_nums)==j+1)
        if len(indexes[0]) ==0 :        #Remove Clusters which have no data points
            K=K-1
            continue
        mean.append(means[j])
        cov.append(covariance(data[indexes],means[j],1,[],[],flag_diag))
        pi_k.append(len(indexes[0])/len(data)) 
    return np.array(mean),np.array(cov),np.array(pi_k),K

#Plot the Distortion as a function of K = 1 to max_K
def plot_Distortion_K(max_K,data,Title=""):
    distortion = []
    for k in range(1,max_K+1):
        distortion.append(0)
        means,cluster_nums,distort = Find_Centroids(k,data,30)
        distortion[-1] = distort[-1]
    plt.plot(np.linspace(1,max_K,max_K),distortion)
    plt.title("Distortion vs No: of Clusters(K): "+Title)
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.show()

#Plot the Distortion as a function of KM num of iterations
def plot_Distortion_num_ite(k,num_ite,data,Title=""):
    means,cluster_nums,distortion = Find_Centroids(k,data,num_ite)
    if len(distortion)<num_ite:
        for i in range(len(distortion),num_ite):
            distortion.append(distortion[-1])
    plt.plot(np.linspace(1,num_ite,num_ite),distortion)
    plt.title("Distortion vs No: of Iterations: "+Title)
    plt.xlabel("No:of Iterations")
    plt.ylabel("Distortion")
    plt.show()

#Computing Covariance Matrices
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

#Computation of Normal_Gaussian 
def Normal_distribution(x,u,E):     
    dim = len(E)
    x_m = x-u
    E_inv = np.linalg.inv(E)
    temp1 = np.dot(x_m,E_inv)
    temp2 = temp1 * x_m
    exp_arg = -1/2 * sum(temp2.T)
    gaus = 1/pow(2*np.pi,dim/2)/np.sqrt(abs(np.linalg.det(E))) * np.exp(exp_arg)
    return gaus

#Finding the Gaussian Mixture Model Parameters
def gmm(data,k,pi_k,mean,cov,num_iterations=10,flag_diag=0):
    log_like_hood = 1e4
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
            if N_k == 0:    #Since contribution of this mixture to any data point is zero, we just ignore it
                continue
            new_mean[j] = sum((data.T*r_nk[:,j]).T)/N_k
            cov[j] = covariance(data,mean[j],0,r_nk[:,j],N_k,flag_diag)
        mean = new_mean
        #Finding the Log Likelihood of data given the current values of parameters
        P = find_prob(1,data,[mean],[cov],[pi_k],[k])
        new_log_like_hood =  sum([np.log10(val) for val in P[:,0]])
        
        #If the old Log-Likehood and the new Log-Likehood values are lesser than the threshold, we end the GMM iterations
        if ite!=0 and (new_log_like_hood-log_like_hood)<5:
            break 
        log_like_hood = new_log_like_hood

    return mean,cov,pi_k

#Finding the Probability that a set of feature vectors belong to each class
def find_prob(no_of_classes,test_data,means,covs,pi,K_s):
    P = np.zeros((len(test_data),no_of_classes))
    for cls in range(no_of_classes):
            numerators = np.zeros((len(test_data),K_s[cls]))
            for j in range(K_s[cls]):
                numerators[:,j] = Normal_distribution(test_data,means[cls][j],covs[cls][j])
            numerators = numerators*pi[cls]
            P[:,cls] = sum(numerators.T)
    return P

#Classifying a feature vector or a set of feature vectors(36 feature vectors for Image_Data_set)
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

#Confusion Matrix
def confusion_matrix(conf_matrix,Title):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title(Title, fontsize=12)
    plt.show()

#ROC and DET
def ROC_DET(S_list,class_labels,Title=""):
    #ROC
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
        plt.plot(FPR,TPR,label="Case "+str(case_no+1))
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("Receiver Operating Characteristic curve "+Title)
    plt.show()

    #DET
    #Makes use of an inbuilt library - sklearn.metrics.det_curve
    #y_true - 1D array of length(no of class * len(dev_data))
    y_true = [0]*(len(S_list[0][0])*len(class_labels))
    count = 0
    for i in range(len(S[0])):
        for j in range(len(class_labels)):
            if class_labels[j] == i+1:
                y_true[count]=1
            count += 1
    plt.figure(figsize=(8,5))  

    #Plotting DET Curve for All Cases
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
    plt.title("Detection Error Tradeoff curve "+Title)
    plt.show()

#Plots the Data Points
#Plots the Decision Boundary
#Plots the Gaussian Mixtures
def contour(class1,class2,means,covs,pi,K_s,priors,flag_diag=0):
    #Creating the 2_D Vector Field
    x1 = np.linspace(-17,17,500)
    x2 = np.linspace(-17,17,500)
    X,Y = np.meshgrid(x1,x2)
    Space = [[] for i in range(len(X)*len(X))]
    count = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            Space[count] = [X[i][j],Y[i][j]]
            count+=1
    Space = np.array(Space)
    
    #Plot the Data
    plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.plot(class1[:,0],class1[:,1],'.',label="Class1  - yellow")
    ax.plot(class2[:,0],class2[:,1],'.',label="Class2 - red")

    #Plotting The Decision Boundary
    P = find_prob(2,Space,means,covs,pi,K_s)
    Z1 = np.reshape(P[:,0],(len(X),len(X)))
    Z2 = np.reshape(P[:,1],(len(X),len(X)))
    classify = 1+np.array([Z1,Z2]).argmax(0)
    c = ax.contourf(X, Y, classify,cmap='YlOrRd',levels=[0,1,2])
    plt.colorbar(c)
    if flag_diag==0:
        ax.set_title("Decision Boundary (Non-Diagonal_Covariance)")
    else:
        ax.set_title("Decision Boundary (Diagonal_Covariance)")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.legend(loc ="lower right")
    Z = [np.zeros((len(X),len(X[0]))),np.zeros((len(X),len(X[0])))]

    #Plotting the Gaussian Mixtures
    for cls in range(2):
        for i in range(K_s[cls]):
            Pdf = Normal_distribution(Space,means[cls][i],covs[cls][i])
            for j in range(len(X)):
                Z[cls][j,:] = Pdf[j*len(X):(j+1)*len(X)]
            ax.contour(X,Y,Z[cls],levels = np.linspace(np.max(Z[cls])/4,np.max(Z[cls]), 10))
    plt.axis('scaled')
    plt.show()

#Image Classification 
def partA(k= 30, km_num_iterations = 10, gmm_num_iterations = 4, flag_diag = 0):
    classes = ['coast','forest','highway','mountain','opencountry']
    train = []  
    train_all = []
    dev = []
    test = []
    denoms = []
    priors = np.zeros(len(classes))

    #Reading the Images of each class for train and Developement DataSets
    for i,cls in enumerate(classes):
        lst = []
        dir_list = os.listdir('Features//'+cls+'//train')
        priors[i] = len(dir_list)
        for file in dir_list:
            lst.extend(np.loadtxt('Features/'+cls+'/train/' + file))
        lst = np.array(lst)
        train.append(lst)
        train_all.extend(lst)

        lst = []
        lst2 = []
        dir_list = os.listdir('Features//'+cls+'//dev')
        for file in dir_list:
            lst.append(np.loadtxt('Features/'+cls+'/dev/' + file))
            lst2.extend(np.loadtxt('Features/'+cls+'/dev/' + file))
        dev.append(np.array(lst))
        test.append(np.array(lst2))

    #Principal Component Analysis, for reducing Dimensionality
    pca = PCA(.99)
    pca.fit(np.array(train_all))
    train_all = pca.transform(train_all)

    #Mean Normalisation
    mean_train = np.mean(train_all,axis=0)
    maxs = np.max(train_all,axis=0)
    mins = np.min(train_all,axis=0)
    denoms = maxs - mins
    train_all = (train_all-mean_train)/denoms

    #Plotting Distortion vs K
    # plot_Distortion_K(30,train[0],"Coast")

    #Plotting Distortion vs No_of_iterations
    plot_Distortion_num_ite(k,50,train[0],"Coast")
    
    #Prior Probs of each Class
    priors = priors/sum(priors)
    means,covs,pi,K_s = [],[],[],[]

    #Training
    for i,clas_data in enumerate(train):
        clas_data =pca.transform(clas_data)
        clas_data = (clas_data-mean_train)/denoms
        mean,cluster_nums,distortion = Find_Centroids(k,clas_data,km_num_iterations)
        print("k_means for class",i+1,": TRAINED")
        mean,cov,pi_k,K = intialization(clas_data,mean,cluster_nums,k,flag_diag)
        new_mean,new_cov,new_pi_k=gmm(clas_data,K,pi_k,mean,cov,gmm_num_iterations,flag_diag)
        print("GMM for class",i+1,": TRAINED")
        means.append(new_mean)
        covs.append(new_cov)
        pi.append(new_pi_k)
        K_s.append(K)
    print("TRAINING COMPLETE")
    print("Testing Dev Data")

    #Test_Evaluation
    count_correct = 0
    count_dev_files = 0
    S = []
    class_labels = []
    conf_matrix = np.zeros((len(train),len(train)))
    for i,test_clas in enumerate(dev):
        _S = [[] for j in range(len(test_clas))]
        class_labels.extend([i+1]*len(test_clas))
        data = test[i]
        data = pca.transform(data)
        data = (data-mean_train)/denoms
        #Finding Probability of feature Vectors
        P = find_prob(len(train),data,means,covs,pi,K_s)

        #Classification of images
        for j in range(len(test_clas)):
            classify2,Probs = classification2(P[j*36:(j+1)*36,:],priors)
            _S[j] = Probs
            count_dev_files +=1
            if classify2 == i:
                count_correct += 1
            conf_matrix[i][classify2] += 1
        S.extend(_S)
        print()
    print("Accuracy = ", count_correct/count_dev_files*100,"%")
    print()
    if flag_diag==0:
        confusion_matrix(conf_matrix,"Image_DataSet_Non_Diagonal_Cov")
    else:
        confusion_matrix(conf_matrix,"Image_DataSet_Diagonal_Cov")
    return S,class_labels

def partB(k= 15, km_num_iterations = 10, gmm_num_iterations = 4, flag_diag=0):
    
    #Reading Train and Dev Data
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
    class1=np.array(class1)
    class2=np.array(class2)
    train.append(class1)
    train.append(class2)

    #Calculating Priors of each Class
    priors = np.zeros(2)
    priors[0] = len(class1)
    priors[1] = len(class2)
    priors = priors/sum(priors)

    
    plt.plot(class1[:,0],class1[:,1],'.',label="class1")
    plt.plot(class2[:,0],class2[:,1],'.',label="class2")
    plt.legend(loc ="lower right")
    plt.title("Synthetic_Dataset")
    plt.show()

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

    #Plotting Distortion vs K
    plot_Distortion_K(30,train[0],"Synthetic: Class1")

    #Plotting Distortion vs No_of_iterations
    # plot_Distortion_num_ite(k,50,train[0],"Synthetic: Class1")

    means,covs,pi,K_s = [],[],[],[]
    #Training
    for i,clas_data in enumerate(train):
        mean,cluster_nums,distortion = Find_Centroids(k,clas_data,km_num_iterations)
        print("k_means for class",i+1,": TRAINED")
        mean,cov,pi_k,K = intialization(clas_data,mean,cluster_nums,k,flag_diag)
        new_mean,new_cov,new_pi_k=gmm(clas_data,K,pi_k,mean,cov,gmm_num_iterations,flag_diag)
        print("GMM for class",i+1,": TRAINED")
        K_s.append(K)
        means.append(new_mean)
        covs.append(new_cov)
        pi.append(new_pi_k)
    
    print("TRAINING COMPLETE")
    print("Testing Dev Data")

    #Contour Plot along with Decision Boundary
    contour(class1,class2,means,covs,pi,K_s,priors,flag_diag)

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

        #Classification of images
        for j in range(len(test_clas)):
            classify2,Probs = classification2(P[j:(j+1),:],priors)
            count_dev_files +=1
            if classify2 == i:
                count_correct += 1
            conf_matrix[i][classify2] += 1
    print("Accuracy = ", count_correct/count_dev_files*100,"%")
    print()
    if flag_diag==0:
        confusion_matrix(conf_matrix,"Synthetic_DataSet_Non_Diagonal_Cov")
    else:
        confusion_matrix(conf_matrix,"Synthetic_DataSet_Diagonal_Cov")
    return(S,class_labels)

S_list = []
S_,class_labels_b = partA(k=15, km_num_iterations = 30, gmm_num_iterations = 4,flag_diag=0)
S_list.append(S_)
S_,class_labels_b = partA(k=15, km_num_iterations = 30, gmm_num_iterations = 4,flag_diag=1)
S_list.append(S_)
ROC_DET(S_list,class_labels_b,"Image_Dataset")

S_list = []
S_,class_labels_b = partB(k=25, km_num_iterations = 13, gmm_num_iterations = 10, flag_diag=0)
S_list.append(S_)
S_,class_labels_b = partB(k=25, km_num_iterations = 13, gmm_num_iterations = 10, flag_diag=1)
S_list.append(S_)
ROC_DET(S_list,class_labels_b,"Synthetic_Dataset")