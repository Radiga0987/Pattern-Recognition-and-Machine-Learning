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
            matrix[i][j] = Euclidean_Distance(aud1[i-1],aud2[j-1]) + \
                            min([matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]])

    return matrix[-1][-1]

predicted_classes=[]

t0=time.time()
for test_digit_idx in range(len(dev)):
    t=time.time()
    model_predictions = []
    for aud_tst in dev[test_digit_idx]:
        min_dist_cls = []
        for train_digit_idx in range(len(train)):
            l=[]
            for aud_ref in train[train_digit_idx]:
                t1=time.time()
                l.append(DTW(aud_ref,aud_tst))
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
# for test_digit_idx in range(len(dev)):
#     t=time.time()
#     model_predictions = []
#     for aud_tst in dev[test_digit_idx]:
#         min_dist_cls = []
#         for train_digit_idx in range(len(train)):
#             l=[]
#             for aud_ref in train[train_digit_idx]:
#                 t1=time.time()
#                 l.append(DTW(aud_ref,aud_tst))
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