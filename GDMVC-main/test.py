import scipy.io
import torch
import logging
import os
import sys
from itertools import chain
from torch import nn

def print_mat_file(mat_file_path):
    # 加载 .mat 文件
    data = scipy.io.loadmat(mat_file_path)

    # 打印文件中的所有变量及其内容
    for key in data:
        print(f"{key}: {data[key]}")
    X = data["X"][0]
    X = [torch.from_numpy(x).float() for x in X]
    a=X[0]
    # print(X)
    print(a)
    print(a.t())


# 替换为你的 .mat 文件的路径
mat_file_path = 'D:\project_clustering\clustering\GDMVC-main\dataset\coil20mv.mat'
print_mat_file(mat_file_path)