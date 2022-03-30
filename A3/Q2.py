import numpy as np
import os


train_1 = []
train_2 = []
train_5 = []
train_9 = []
train_z = []
dev_1 = []

dir_list = os.listdir('Isolated_Digits//1//train')
for file in dir_list:
    if file.endswith('.mfcc'):
        with open(r'Isolated_Digits/1/train/'+file) as f:
            a = f.read().splitlines()
        for i in range(len(a)):
            a[i] = np.array([float(s) for s in a[i].split()])

        train_1.append(np.array(a[1:]))

dir_list = os.listdir('Isolated_Digits//2//train')
for file in dir_list:
    if file.endswith('.mfcc'):
        with open(r'Isolated_Digits/2/train/'+file) as f:
            a = f.read().splitlines()
        for i in range(len(a)):
            a[i] = np.array([float(s) for s in a[i].split()])

        train_2.append(np.array(a[1:]))

dir_list = os.listdir('Isolated_Digits//5//train')
for file in dir_list:
    if file.endswith('.mfcc'):
        with open(r'Isolated_Digits/5/train/'+file) as f:
            a = f.read().splitlines()
        for i in range(len(a)):
            a[i] = np.array([float(s) for s in a[i].split()])

        train_5.append(np.array(a[1:]))

dir_list = os.listdir('Isolated_Digits//9//train')
for file in dir_list:
    if file.endswith('.mfcc'):
        with open(r'Isolated_Digits/9/train/'+file) as f:
            a = f.read().splitlines()
        for i in range(len(a)):
            a[i] = np.array([float(s) for s in a[i].split()])

        train_9.append(np.array(a[1:]))

dir_list = os.listdir('Isolated_Digits//z//train')
for file in dir_list:
    if file.endswith('.mfcc'):
        with open(r'Isolated_Digits/z/train/'+file) as f:
            a = f.read().splitlines()
        for i in range(len(a)):
            a[i] = np.array([float(s) for s in a[i].split()])

        train_z.append(np.array(a[1:]))
print("lol")
dir_list = os.listdir('Isolated_Digits//1//dev')
for file in dir_list:
    if file.endswith('.mfcc'):
        with open(r'Isolated_Digits/1/dev/'+file) as f:
            a = f.read().splitlines()
        for i in range(len(a)):
            a[i] = np.array([float(s) for s in a[i].split()])

        dev_1.append(np.array(a[1:]))

def Euclidean_Distance(v1,v2):
    d = np.sqrt(sum((v1-v2)**2))
    return(d)

def DTW(aud1,aud2):
    n,m = len(aud1),len(aud2)
    matrix = np.zeros((n+1,m+1))

    for i in range(n+1):
        matrix[i][0] = float("inf")

    for j in range(m+1):
        matrix[0][j] = float("inf")

    matrix[0,0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            matrix[i,j] = Euclidean_Distance(aud1[i-1],aud2[j-1]) + \
                            min([matrix[i-1, j], matrix[i, j-1], matrix[i-1, j-1]])

    return matrix[-1,-1]

classes=[]
for aud_tst in [dev_1[0]]:
    min_dist_cls = []
    for cls in [train_1,train_2,train_5,train_9,train_z]:
        min_val = float("inf")
        for aud_ref in cls:
            min_val = min(min_val,DTW(aud_ref,aud_tst))
        min_dist_cls.append(min_val)

    class_idx = 0
    min_class_val=float("inf")
    for i in range(len(min_dist_cls)):
        if min_class_val > min_dist_cls[i]:
            min_class_val = min_dist_cls[i]
            class_idx = i

    classes.append(class_idx)

print(classes)
print("Accuracy = ",classes.count(0)/len(classes))

