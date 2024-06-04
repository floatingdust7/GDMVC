import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as compute_nmi
from sklearn.metrics.cluster._supervised import check_clusterings

'''
这段代码是一个Python脚本，用于执行聚类分析，特别是使用K-Means算法。它包括了多个函数，用于计算聚类结果、评估聚类质量，并记录日志。下面是对每个函数的分析：

get_cluster_result(features, n_clusters):

这个函数使用KMeans算法从sklearn.cluster模块对给定的特征进行聚类。
features是待聚类的数据点的特征矩阵。
n_clusters是聚类的数量。
函数返回KMeans算法预测的每个数据点的聚类标签。
compute_acc(Y, Y_pred):

这个函数用于计算聚类的准确度。
Y是真实的标签。
Y_pred是预测的标签。
它使用匈牙利算法（通过linear_sum_assignment函数）来找到最优的标签匹配，并计算准确度。
compute_fscore(labels_true, labels_pred):

这个函数计算F-score，它是精度和召回率的调和平均。
labels_true是真实的聚类标签。
labels_pred是预测的聚类标签。
函数首先创建每个真实和预测聚类中样本的集合，然后计算精度和召回率，最后根据这些值计算F-score。
cluster_one_time(features, labels, n_clusters):

这个函数执行单次聚类，并计算几种不同的聚类评估指标。
它首先调用get_cluster_result来获取聚类结果。
然后，它调用compute_nmi、compute_acc和compute_fscore来计算归一化互信息(NMI)、准确度(ACC)和F-score。
函数返回这些指标的值，并进行四舍五入。
cluster(n_clusters, features, labels, count=1, desc="cluster_mine"):

这个函数用于多次执行聚类，并计算评估指标的平均值和标准差。
n_clusters是聚类的数量。
features是待聚类的数据点的特征矩阵。
labels是真实的聚类标签。
count是聚类执行的次数，默认为1。
desc是一个描述性字符串，用于日志记录。
函数循环count次，每次调用cluster_one_time，并记录结果。
最后，它计算并记录NMI、ACC和F-score的平均值和标准差。
整个脚本的目的是提供一个工具，用于评估K-Means聚类算法在特定数据集上的性能。它通过计算不同的评估指标（NMI、ACC和F-score）来衡量聚类的质量，并提供了一种方法来评估算法的一致性和可靠性，通过多次运行聚类并计算结果的统计数据。此外，脚本还包括了日志记录功能，这对于调试和跟踪实验结果非常有用。
'''
def get_cluster_result(features, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init=10)
    pred = km.fit_predict(features)
    return pred


def compute_acc(Y, Y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    from scipy.optimize import linear_sum_assignment

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size * 100


def compute_fscore(labels_true, labels_pred):
    # b3_precision_recall_fscore就是Fscore
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError("input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score


def cluster_one_time(features, labels, n_clusters):
    pred = get_cluster_result(features, n_clusters)
    labels = np.reshape(labels, np.shape(pred))
    if np.min(labels) == 1:
        labels -= 1

    nmi = compute_nmi(labels, pred) * 100
    acc = compute_acc(labels, pred)
    fscore = compute_fscore(labels, pred) * 100
    return round(nmi, 2), round(acc, 2), round(fscore, 2)


def cluster(n_clusters, features, labels, count=1, desc="cluster_mine"):
    nmi_array, acc_array, f1_array = [], [], []
    for _ in range(count):
        nmi, acc, fscore = cluster_one_time(features, labels, n_clusters)
        nmi_array.append(nmi)
        acc_array.append(acc)
        f1_array.append(fscore)
        logging.info(f"kmeans NMI={nmi}, ACC={acc}, Fscore={fscore}")

    nmi_avg, nmi_std = round(np.mean(nmi_array), 2), round(np.std(nmi_array), 2)
    acc_avg, acc_std = round(np.mean(acc_array), 2), round(np.std(acc_array), 2)
    f1_avg, f1_std = round(np.mean(f1_array), 2), round(np.std(f1_array), 2)
    logging.info(
        f"{desc} Kmeans({count} times average) NMI={nmi_avg:.2f}±{nmi_std:.2f}, ACC={acc_avg:.2f}±{acc_std:.2f}, fscore={f1_avg:.2f}±{f1_std:.2f}"
    )
    results=[nmi_avg, nmi_std, acc_avg, acc_std, f1_avg, f1_std]
    return results
