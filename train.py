import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np
import time

from dataset import *
from trainset import *

def run_stat(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-time independent runs"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each run
    all_scores = []
    data = get_data(path=args.data_path, args=args, logger=logger)  ####在这里就定义了数据的内容
    for run_num in range(args.num_runs):
        info(f'Run {run_num}')
        args.seed = init_seed + run_num
        set_seed(args.seed)
        args.save_dir = os.path.join(save_dir, f'run_{run_num}')
        makedirs(args.save_dir)
        model_scores = a_run_training(args,data, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    #打印整个结果
    info(f'{args.num_runs}-time runs')
    for run_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + run_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

    #打印最终结果
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')


    return mean_score, std_score


if __name__ == '__main__':
    args = parse_train_args()
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    mean_auc_score, std_auc_score = run_stat(args, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')