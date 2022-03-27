import numpy as np
# def cal(v):
#     return sum(v)

data = np.array([[1,2,3],[2,4,6]])
pro = np.array([1,2])
# pi_k = [1/2,1/4]
# k = 2
# r_nk = np.zeros((2,2))
# for i,x in enumerate(data):
#         numerators = np.array([pi_k[j] * cal(x) for j in range(k)])
#         print(numerators)
#         denominators = sum(numerators)
#         print(denominators)
#         r_nk[i,:] = numerators/denominators
# print(r_nk)
print(data.T *pro)