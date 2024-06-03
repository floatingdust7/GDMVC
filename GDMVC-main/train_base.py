import logging
import os
import torch
import sys
from itertools import chain
from torch import nn

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "code"))

from data import load_dataset
from model import AdaGAEMV
from utils import update_graph, cluster_by_multi_ways


def train_base(
    dataset_name, args, is_finetune=True, is_fusion=True, is_graph=True, fixed_k=None
):
    """基础训练函数，其他训练函数均由该函数修改得到"""
    #记录关键的训练参数，包括数据集名称 dataset_name 和参数字典 args。
    logging.info(f"dataset: {dataset_name}, args: {args}")
    # 记录布尔参数is_finetune、is_fusion和is_graph，这些参数指示训练过程中是否执行微调、融合和图卷积网络的使用。
    logging.info(
        f"is_finetune: {is_finetune}, is_fusion: {is_fusion}, is_graph: {is_graph}"
    )

    # 根据is_fusion参数的值，设置fusion_kind变量
    fusion_kind = "pinjiezv_pingjunlv_lxz" if is_fusion else "pinjiezv"
    # 根据is_graph参数的值，从args字典中获取lam_tr（正则化系数），如果不使用图卷积网络，则将其设置为0。
    lam_tr = args["lam_tr"] if is_graph else 0

    logging.info("load data")
    #调用 load_dataset 函数，传入 dataset_name，以获取数据集 X、标签 Y、聚类数量 n_cluster、样本数 n_sample 和视图数 n_view。
    X, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #将数据集 X 中的数据转换为 PyTorch 张量，并将它们移动到设置的计算设备上。这里假设 X 中的数据是从 NumPy 数组加载的，并且需要转换为浮点数类型。
    X = [torch.from_numpy(x).float().to(device) for x in X]

    logging.info("compute similarity")
    neighbor_num = args["neighbor_init"] if fixed_k is None else fixed_k
    weights_mv, raw_weights_mv, laplacian_mv = update_graph(X, neighbor_num)

    # cluster_by_multi_ways(X, laplacian_mv, Y, n_cluster, fusion_kind="pinjiezv")

    logging.info("init model and optimizer")
    gae_model = AdaGAEMV(X, args["layers"], device)
    optimizer = torch.optim.Adam(
        chain(*[gae_model.gae_list[v].parameters() for v in range(n_view)]),
        lr=args["learning_rate"],
    )

    logging.info("start pretrain")
    for epoch in range(args["pretrain_epoch"]):
        logging.info(
            f"neighbor_num: {neighbor_num}, neighbor_max: {args['neighbor_max']}"
        )

        for i in range(args["pretrain_iter"]):
            embedding_list, recons_w_list = gae_model.forward(X, laplacian_mv)
            re_loss, tr_loss = gae_model.cal_loss(
                raw_weights_mv,
                recons_w_list,
                weights_mv,
                embedding_list,
                lam_tr,
            )
            loss = re_loss + lam_tr * tr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % args["log_freq"] == 0:
                logging.info(
                    f'Epoch[{epoch+1}/{args["pretrain_epoch"]}], Step [{i+1}/{args["pretrain_iter"]}], L: {loss.item():.4f}, Lre: {re_loss.item():.4f}, Ltr: {tr_loss.item():.8f}'
                )

        weights_mv, raw_weights_mv, laplacian_mv = update_graph(
            embedding_list, neighbor_num
        )
        if fixed_k is None:
            neighbor_num = min(
                neighbor_num + args["neighbor_incr"], args["neighbor_max"]
            )

        # cluster_by_multi_ways(embedding_list, laplacian_mv, Y, n_cluster)

    if is_finetune:
        logging.info("start fine tuning")
        mse_loss_func = nn.MSELoss()
        for epoch in range(args["finetune_epoch"]):
            embedding_list, recons_w_list = gae_model.forward(X, laplacian_mv)
            re_loss, tr_loss = gae_model.cal_loss(
                raw_weights_mv,
                recons_w_list,
                weights_mv,
                embedding_list,
                lam_tr,
            )
            con_loss = 0
            for vi in range(n_view):
                for vj in range(vi + 1, n_view):
                    con_loss += mse_loss_func(embedding_list[vi], embedding_list[vj])
            loss = re_loss + lam_tr * tr_loss + args["lam_con"] * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % args["log_freq"] == 0:
                logging.info(
                    f'Epoch[{epoch+1}/{args["finetune_epoch"]}], L: {loss.item():.4f}, Lre: {re_loss.item():.4f}, Ltr: {tr_loss.item():.4f}, Lcon: {con_loss.item():.4f}'
                )
            # if (epoch + 1) % 10 == 0:
            #     cluster_by_multi_ways(embedding_list, laplacian_mv, Y, n_cluster)

    results = cluster_by_multi_ways(
        embedding_list, laplacian_mv, Y, n_cluster, count=10, fusion_kind=fusion_kind
    )
    return results
