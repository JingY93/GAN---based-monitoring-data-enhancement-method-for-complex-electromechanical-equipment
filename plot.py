# coding:UTF-8
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import DTW
import mmd
import torch
# 画原图
data = scio.loadmat('raw.mat')
data=data['nor']
data=data[0:487424].reshape(-1,64,64)
# plt.subplot(2,4,1)
# plt.imshow(data[100], cmap="gray")
# plt.plot(data[0:64*64])
# plt.subplot(2,4,2)
# plt.imshow(data[1], cmap="gray")
# plt.subplot(2,4,3)
# plt.imshow(data[2], cmap="gray")
# plt.subplot(2,4,4)
# plt.imshow(data[3], cmap="gray")
# # plt.show()
# # np.save('bearing_data',data)
# # 画新图
A=np.load('generated_100.npy').reshape(-1,64,64)
# plt.subplot(2,4,5)
# plt.imshow(A[10], cmap="gray")
# plt.subplot(2,4,6)
# plt.imshow(A[11], cmap="gray")
# plt.subplot(2,4,7)
# plt.imshow(A[12], cmap="gray")
# plt.subplot(2,4,8)
# plt.imshow(A[13], cmap="gray")
# plt.savefig('1')
plt.show()

# DTW验证
# s1 = data[100,:,:].flatten()
# s2 = data[50,:,:].flatten()
# s3 = A[1,:,:].flatten()

# # 原始算法
# distance12, paths12, max_sub12 = DTW.TimeSeriesSimilarityImprove(s1, s2)
# distance13, paths13, max_sub13 = DTW.TimeSeriesSimilarityImprove(s1, s3)
#
# print("更新前s1和s2距离：" + str(distance12))
# print("更新前s1和s3距离：" + str(distance13))
#
# # 衰减系数
# weight12 = DTW.calculate_attenuate_weight(len(s1), len(s2), max_sub12)
# weight13 = DTW.calculate_attenuate_weight(len(s1), len(s3), max_sub13)
#
# # 更新距离
# print("更新后s1和s2距离：" + str(distance12 * weight12))
# print("更新后s1和s3距离：" + str(distance13 * weight13))

# MMD验证
s1 =torch.from_numpy(data[100,:,:])
s2 = torch.from_numpy(data[50,:,:])
s3 = torch.from_numpy(A[10,:,:])
dict1=mmd.mmd_rbf(s1,s2)
dict2=mmd.mmd_rbf(s1,s3)
print("s1和s2MMD距离：" + str(dict1))
print("s1和s3MMD距离：" + str(dict2))