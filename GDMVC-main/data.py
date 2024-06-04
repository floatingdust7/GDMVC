import logging
import os
import numpy as np
from sklearn import preprocessing
import scipy.io as sio
import scipy.sparse


def load_new_format_data(dataset_name):
    #获取当前执行文件所在的路径
    file_dir =os.path.dirname(__file__)
    data_path = f"{file_dir}/dataset/{dataset_name}.mat"
    data = sio.loadmat(data_path)
    X = data["X"][0]
    Y = data["Y"]
    # print_multi_view_data(X, Y)
    return X, Y


def normalize_feature(X, dataset_name):
    if isinstance(X[0], scipy.sparse.csc_matrix):
        # 每个视角的数据类型为<class 'scipy.sparse._csc.csc_matrix'>稀疏矩阵，不能进行中心化
        #但是进行标准化了，方差被调为1
        X = [preprocessing.scale(x, with_mean=False) for x in X]
    else:
        if dataset_name in ["coil20mv", "hdigit"]:
            # 这两个数据集用scale方法处理，报如下警告：UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
            scaler = preprocessing.StandardScaler()
            X = [scaler.fit_transform(x) for x in X]
        else:
            X = [preprocessing.scale(x) for x in X]
    return X


def load_dataset(dataset_name):
    X, Y = load_new_format_data(dataset_name)
    #将标签数据 Y 的数据类型转换为整数类型
    Y = Y.astype(int)
    #计算类别数
    n_cluster = len(np.unique(Y))
    #计算样本数
    n_sample = len(Y)
    #计算视图数量
    n_view = len(X)

    X = normalize_feature(X, dataset_name)
    logging.info(
        f"dataset:{dataset_name}, n_cluster: {n_cluster}, n_sample: {n_sample}, n_view: {n_view}"
    )
    return X, Y, n_cluster, n_sample, n_view
