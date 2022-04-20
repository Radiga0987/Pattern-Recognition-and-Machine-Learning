import numpy as np
import os
import time

digits = [1,2,5,9,'z']
train = []
dev = []

for digit in digits:
    lst = []
    dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//train')
    for file in dir_list:
        if file.endswith('.mfcc'):
            lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/train/' + file,skiprows = 1))

    train.append(lst)

    lst = []
    dir_list = os.listdir('Isolated_Digits//'+str(digit)+'//dev')
    for file in dir_list:
        if file.endswith('.mfcc'):
            lst.append(np.loadtxt('Isolated_Digits/'+str(digit)+'/dev/' + file,skiprows = 1))

    dev.append(lst)

def Euclidean_Distance(v1,v2):
    d = np.linalg.norm(v1-v2)
    return(d)

def DTW(aud1,aud2):
    n,m = len(aud1),len(aud2)
    matrix = [[0 for j in range(m+1)] for i in range(n+1)]

    for i in range(n+1):
        matrix[i][0] = float("inf")

    for j in range(m+1):
        matrix[0][j] = float("inf")

    matrix[0][0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            matrix[i][j] = Euclidean_Distance(aud1[i-1],aud2[j-1]) + min([matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]])

    return matrix[-1][-1]

def DTW_Spoken_Digits():
    predicted_classes=[]

    t0=time.time()
    for test_cls in range(len(dev)):
        t=time.time()
        model_predictions = []
        for tst_sample in [dev[test_cls][0]]:
            min_dist_cls = []
            for trn_cls in range(len(train)):
                l=[]
                for trn_sample in train[trn_cls]:
                    t1=time.time()
                    l.append(DTW(trn_sample,tst_sample))
                    t2=time.time()
                    print(t2-t1)
                min_dist_cls.append(min(l))

            model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
        predicted_classes.append(model_predictions)
        t11=time.time()
        print("lolololol",t11-t)
    tf=time.time()
    print(tf-t0)

    # t0=time.time()
    # for test_cls in range(len(dev)):
    #     t=time.time()
    #     model_predictions = []
    #     for tst_sample in dev[test_cls]:
    #         min_dist_cls = []
    #         for trn_cls in range(len(train)):
    #             l=[]
    #             for trn_sample in train[trn_cls]:
    #                 t1=time.time()
    #                 l.append(DTW(trn_sample,tst_sample))
    #                 t2=time.time()
    #                 print(t2-t1)
    #             min_dist_cls.append(sum(l)/len(l))

    #         model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
    #     predicted_classes.append(model_predictions)
    #     t11=time.time()
    #     print("lolololol",t11-t)
    # tf=time.time()
    # print(tf-t0)

    print(predicted_classes)

    count_correct = 0
    count_dev_files = 0
    for i in range(len(predicted_classes)):
        for j in predicted_classes[i]:
            count_dev_files +=1
            if j == i:
                count_correct += 1

    print("Accuracy = ", count_correct/count_dev_files)
    print(count_dev_files)





###########################################################################
letters = ['a','bA','chA','lA','tA']
train = []
dev = []

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
        train_lst[len(train_lst) - 1].append(train_lst[len(train_lst) - 2][2])

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
        test_lst[len(test_lst) - 1].append(test_lst[len(test_lst) - 2][2])

        lst.append(np.array(test_lst))

    dev.append(lst)

def DTW_Telugu_Chars():
    predicted_classes=[]

    t0=time.time()
    for test_cls in range(len(dev)):
        t=time.time()
        model_predictions = []
        for tst_sample in dev[test_cls]:
            min_dist_cls = []
            for trn_cls in range(len(train)):
                l=[]
                for trn_sample in train[trn_cls]:
                    t1=time.time()
                    l.append(DTW(trn_sample,tst_sample))
                    t2=time.time()
                    print(t2-t1)
                min_dist_cls.append(min(l))

            model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
        predicted_classes.append(model_predictions)
        t11=time.time()
        print("lolololol",t11-t)
    tf=time.time()
    print(tf-t0)

    print(predicted_classes)

    count_correct = 0
    count_dev_files = 0
    for i in range(len(predicted_classes)):
        for j in predicted_classes[i]:
            count_dev_files +=1
            if j == i:
                count_correct += 1

    print("Accuracy = ", count_correct/count_dev_files)
    print(count_dev_files)




DTW_Telugu_Chars()












"""def DTW_Telugu_Chars():
    predicted_classes=[]

    t0=time.time()
    for test_cls in range(len(dev)):
        t=time.time()
        model_predictions = []
        for tst_sample in dev[test_cls]:
            min_dist_cls = []
            for trn_cls in range(len(train)):
                l=[]
                for trn_sample in train[trn_cls]:
                    t1=time.time()
                    l.append(DTW(trn_sample,tst_sample))
                    t2=time.time()
                    print(t2-t1)
                min_dist_cls.append(min(l))

            model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
        predicted_classes.append(model_predictions)
        t11=time.time()
        print("lolololol",t11-t)
    tf=time.time()
    print(tf-t0)

    print(predicted_classes)

    count_correct = 0
    count_dev_files = 0
    for i in range(len(predicted_classes)):
        for j in predicted_classes[i]:
            count_dev_files +=1
            if j == i:
                count_correct += 1

    print("Accuracy = ", count_correct/count_dev_files)
    print(count_dev_files)"""








"""def DTW_Spoken_Digits():
    predicted_classes=[]

    t0=time.time()
    for test_cls in range(len(dev)):
        t=time.time()
        model_predictions = []
        for tst_sample in [dev[test_cls][0]]:
            min_dist_cls = []
            for trn_cls in range(len(train)):
                l=[]
                for trn_sample in train[trn_cls]:
                    t1=time.time()
                    l.append(DTW(trn_sample,tst_sample))
                    t2=time.time()
                    print(t2-t1)
                min_dist_cls.append(min(l))

            model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
        predicted_classes.append(model_predictions)
        t11=time.time()
        print("lolololol",t11-t)
    tf=time.time()
    print(tf-t0)

    # t0=time.time()
    # for test_cls in range(len(dev)):
    #     t=time.time()
    #     model_predictions = []
    #     for tst_sample in dev[test_cls]:
    #         min_dist_cls = []
    #         for trn_cls in range(len(train)):
    #             l=[]
    #             for trn_sample in train[trn_cls]:
    #                 t1=time.time()
    #                 l.append(DTW(trn_sample,tst_sample))
    #                 t2=time.time()
    #                 print(t2-t1)
    #             min_dist_cls.append(sum(l)/len(l))

    #         model_predictions.append(min_dist_cls.index(min(min_dist_cls)))
    #     predicted_classes.append(model_predictions)
    #     t11=time.time()
    #     print("lolololol",t11-t)
    # tf=time.time()
    # print(tf-t0)

    print(predicted_classes)

    count_correct = 0
    count_dev_files = 0
    for i in range(len(predicted_classes)):
        for j in predicted_classes[i]:
            count_dev_files +=1
            if j == i:
                count_correct += 1

    print("Accuracy = ", count_correct/count_dev_files)
    print(count_dev_files)"""
