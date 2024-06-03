import logging
import os

from train_base import train_base
from utils import prepare_log, train_wrapper
from constant import dataset_list, fixed_args, dataset_args_map


def train_duibi(dataset_name, args):
    return train_base(dataset_name, args)


if __name__ == "__main__":
    #train_kind 可以被设置为不同的值，以指向不同的训练函数，从而允许同一个训练脚本支持多种训练模式或算法。这种设计使得脚本更加通用和可扩展。
    train_kind = "train_duibi"
    view_index = 0

    #日志存放的目录
    log_dir = os.path.join(os.path.dirname(__file__), "output-log")
    #日志名称
    log_file_name = f"{train_kind}"
    prepare_log(3, log_dir, log_file_name)

    recorder = [["dataset", "results"]]
    #这段代码的目的是自动化地对dataset_list中的每个数据集进行训练，记录训练进度，并收集每个数据集的训练结果。
    for dataset_index in range(len(dataset_list)):
        dataset_name = dataset_list[dataset_index]
        logging.info(
            f"duibi progress: dataset_name[{dataset_index+1}/{len(dataset_list)}]"
        )

        args = dataset_args_map[dataset_name]
        args.update(fixed_args)
        #train_wrapper调用函数上面def的函数，也就调用了train_base
        results = train_wrapper(train_duibi, dataset_name, args, gpu_id=0)
        recorder.append([dataset_name, results])
        # break
    #每条日志都记录了一个数据集的名称和它的训练结果。
    for row in recorder:
        logging.info(row)
