import numpy as np
import os

#Same K-means code from K-means GMM part of the assignment 
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


#GENERATING THE CODEBOOK AND HENCE THE SYMBOL SEQUENCES FOR TRAIN AND TEST OF ISOLATED DIGITS
#Performing K means on all the features of all classes
digits = [1,2,5,9,'z']
train = []
train_all = []
dev = []
dev_all= []

for digit in digits:
    lst= []
    dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//train')
    for file in dir_list:
        if file.endswith('.mfcc'):
            lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/train/' + file,skiprows = 1))
            train_all.extend(np.loadtxt('Isolated_Digits/'+str(digit)+'/train/' + file,skiprows = 1))
    train.append(lst)

    lst = []
    dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//dev')
    for file in dir_list:
        if file.endswith('.mfcc'):
            lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/dev/' + file,skiprows = 1))
            dev_all.extend(np.loadtxt('Isolated_Digits/'+str(digit)+'/dev/' + file,skiprows = 1))
    dev.append(lst)

k = 44    #Number of clusters 
means , cluster_nums = Find_Centroids(k,train_all,40)

symbols_classes = []
count = 0
for cls in range(len(digits)):
    lst= []
    for file in train[cls]:
        symbols = []
        for j in range(len(file)):
            symbols.append(cluster_nums[count]-1)
            count += 1
        lst.append(symbols)
    symbols_classes.append(lst)

for cls in range(len(digits)):
    with open('Digits_Symbol_Data/train/train_sequence'+str(cls+1)+'_digits.hmm.seq', 'w') as f:
        pass
    with open('Digits_Symbol_Data/train/train_sequence'+str(cls+1)+'_digits.hmm.seq', 'a') as f:
        for symbols in symbols_classes[cls]:
            f.write(' '.join(str(symbol) for symbol in symbols))
            f.write('\n')

cluster_nums_dev = Assign_Cluster(k,np.array(dev_all),means)
symbols_classes = []
count = 0
for cls in range(len(digits)):
    lst= []
    for file in dev[cls]:
        symbols = []
        for j in range(len(file)):
            symbols.append(cluster_nums_dev[count]-1)
            count += 1
        lst.append(symbols)
    symbols_classes.append(lst)

for cls in range(len(digits)):
    with open('Digits_Symbol_Data/dev/test_sequence'+str(cls+1)+'_digits.hmm.seq', 'w') as f:
        pass
    with open('Digits_Symbol_Data/dev/test_sequence'+str(cls+1)+'_digits.hmm.seq', 'a') as f:
        for symbols in symbols_classes[cls]:
            f.write(' '.join(str(symbol) for symbol in symbols))
            f.write('\n')








#GENERATING THE CODEBOOK AND HENCE THE SYMBOL SEQUENCES FOR TRAIN AND TEST OF HANDWRITING DATA
letters = ['a','bA','chA','lA','tA']
train = []
train_all = []
dev = []
dev_all= []

for letter in letters:
    lst = []
    dir_list = os.listdir('Handwriting_Data//'+letter+'//train')
    for file in dir_list:
        temp = np.loadtxt('Handwriting_Data/'+letter+'/train/' + file)[1:]
        train_lst = np.array([temp[::2],temp[1::2]]).T
        train_lst = train_lst.tolist()
        for i in range(len(train_lst)-1):
            if train_lst[i+1][0] != train_lst[i][0]: 
                train_lst[i].append(np.arctan((train_lst[i+1][1] - train_lst[i][1])/(train_lst[i+1][0] - train_lst[i][0])))
            else:
                if train_lst[i+1][1] - train_lst[i][1] > 0:
                    train_lst[i].append(np.arctan(np.inf))
                else:
                    train_lst[i].append(np.arctan(-np.inf))
            train_all.append(np.array(train_lst[i]))
        train_lst[len(train_lst) - 1].append(train_lst[len(train_lst) - 2][2])
        train_all.append(np.array(train_lst[-1]))

        lst.append(np.array(train_lst))

    train.append(lst)

    lst = []
    dir_list = os.listdir('Handwriting_Data//'+letter+'//dev')
    for file in dir_list:
        temp = np.loadtxt('Handwriting_Data/'+letter+'/dev/' + file)[1:]
        test_lst = np.array([temp[::2],temp[1::2]]).T
        test_lst = test_lst.tolist()
        for i in range(len(test_lst)-1):
            if test_lst[i+1][0] != test_lst[i][0]: 
                test_lst[i].append(np.arctan((test_lst[i+1][1] - test_lst[i][1])/(test_lst[i+1][0] - test_lst[i][0])))
            else:
                if test_lst[i+1][1] - test_lst[i][1] > 0:
                    test_lst[i].append(np.arctan(np.inf))
                else:
                    test_lst[i].append(np.arctan(-np.inf))
            dev_all.append(np.array(test_lst[i]))
        test_lst[len(test_lst) - 1].append(test_lst[len(test_lst) - 2][2])
        dev_all.append(np.array(test_lst[-1]))

        lst.append(np.array(test_lst))

    dev.append(lst)

#Performing K means on all the features of all classes
k = 10   #Number of clusters 
means , cluster_nums = Find_Centroids(k,train_all,40)

symbols_classes = []
count = 0
for cls in range(len(letters)):
    lst= []
    for file in train[cls]:
        symbols = []
        for j in range(len(file)):
            symbols.append(cluster_nums[count]-1)
            count += 1
        lst.append(symbols)
    symbols_classes.append(lst)

for cls in range(len(digits)):
    with open('Letters_Symbol_Data/train/train_sequence'+str(cls+1)+'_letters.hmm.seq', 'w') as f:
        pass
    with open('Letters_Symbol_Data/train/train_sequence'+str(cls+1)+'_letters.hmm.seq', 'a') as f:
        for symbols in symbols_classes[cls]:
            f.write(' '.join(str(symbol) for symbol in symbols))
            f.write('\n')

cluster_nums_dev = Assign_Cluster(k,np.array(dev_all),means)
symbols_classes = []
count = 0
for cls in range(len(digits)):
    lst= []
    for file in dev[cls]:
        symbols = []
        for j in range(len(file)):
            symbols.append(cluster_nums_dev[count]-1)
            count += 1
        lst.append(symbols)
    symbols_classes.append(lst)

for cls in range(len(digits)):
    with open('Letters_Symbol_Data/dev/test_sequence'+str(cls+1)+'_letters.hmm.seq', 'w') as f:
        pass
    with open('Letters_Symbol_Data/dev/test_sequence'+str(cls+1)+'_letters.hmm.seq', 'a') as f:
        for symbols in symbols_classes[cls]:
            f.write(' '.join(str(symbol) for symbol in symbols))
            f.write('\n')