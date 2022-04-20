from inspect import trace
import numpy as np
import os
# # def cal(v):
# #     return sum(v)

# data = np.array([[1,2,3],[2,4,6]])
# pro = np.array([1,2])
# # pi_k = [1/2,1/4]
# # k = 2
# # r_nk = np.zeros((2,2))
# # for i,x in enumerate(data):
# #         numerators = np.array([pi_k[j] * cal(x) for j in range(k)])
# #         print(numerators)
# #         denominators = sum(numerators)
# #         print(denominators)
# #         r_nk[i,:] = numerators/denominators
# # print(r_nk)
# print(data.T *pro)

# dir_list = os.listdir('Isolated_Digits//1//train')
# train_1 = []

# for file in dir_list:
#     if file.endswith('.mfcc'):
#         with open(r'Isolated_Digits/1/train/'+file) as f:
#             a = f.read().splitlines()
#         for i in range(len(a)):
#             a[i] = [float(s) for s in a[i].split()]

#         train_1.append(a)

# list = np.array([1,2,3,4,5,6]) 
# print(np.array([list[::2],list[1::2]]).T)
# print(list[::2][0])


# for letter in letters:
#     lst = []
#     dir_list = os.listdir('Handwriting_Data//'+letter+'//train')
#     for file in dir_list:
#         temp = np.loadtxt('Handwriting_Data/'+letter+'/train/' + file)[1:]
#         train_lst = np.array([temp[::2],temp[1::2]]).T
#         for i in range(len(train_lst)-1):
#             if train_lst[i+1][0] != train_lst[i][0]: 
#                 train_lst[i] = np.append(train_lst[i] , (train_lst[i+1][1] - train_lst[i][1])/(train_lst[i+1][0] - train_lst[i][0]))
#             else:
#                 train_lst[i] = np.append(train_lst[i] , MAX_INT)
#         train_lst[len(train_lst) - 1] = np.append(train_lst[len(train_lst) - 1] , train_lst[len(train_lst) - 2][2])

#         lst.append(train_lst)

#     train.append(lst)

#     lst = []
#     dir_list = os.listdir('Handwriting_Data//'+letter+'//dev')
#     for file in dir_list:
#         temp = np.loadtxt('Handwriting_Data/'+letter+'/dev/' + file)[1:]
#         test_lst = np.array([temp[::2],temp[1::2]]).T
#         for i in range(len(test_lst)-1):
#             if test_lst[i+1][0] != test_lst[i][0]:
#                 test_lst[i] = np.append(test_lst[i] , (test_lst[i+1][1] - test_lst[i][1])/(test_lst[i+1][0] - test_lst[i][0]))
#             else:
#                 test_lst[i] = np.append(test_lst[i] , MAX_INT)
#         test_lst[len(test_lst) - 1] = np.append(test_lst[len(test_lst) - 1] , test_lst[len(test_lst) - 2][2])
#         lst.append(test_lst)

#     dev.append(lst)

print([1,2,3][:-2])