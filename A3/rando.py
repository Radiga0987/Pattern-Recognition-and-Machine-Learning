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

# a = 2*np.pi
# print(pow(2*np.pi,3/2))
# print(np.sqrt(a**3))

a = np.array([[1,2,3]])
b = np.array([[2,1,3],[8,4,5]])
print(np.argmax(b[1]))

a = np.array([[1,25,3],[17,2,43]])
print(np.max(a,axis=0))