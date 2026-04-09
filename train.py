import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
import json
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
    run_records = []
    data = get_data(path=args.data_path, args=args, logger=logger)  
    for run_num in range(args.num_runs):
        args.seed = init_seed + run_num
        info(f'Run {run_num} | Seed {args.seed}')
        set_seed(args.seed)
        # Save each seed in an explicit seed folder so checkpoints are easy to audit.
        args.save_dir = os.path.join(save_dir, f'seed_{args.seed}')
        makedirs(args.save_dir)
        model_scores = a_run_training(args,data, logger)
        all_scores.append(model_scores)
        model_path = os.path.join(args.save_dir, 'model.pt')
        run_records.append({
            'run_num': run_num,
            'seed': args.seed,
            'avg_test_score': float(np.nanmean(model_scores)),
            'model_path': model_path
        })
        info(f'Run {run_num} checkpoint: {model_path}')
    all_scores = np.array(all_scores)

    # Persist per-seed summary for later model selection and audit.
    summary_path = os.path.join(save_dir, 'seed_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(run_records, f, indent=2)
    info(f'Saved seed summary: {summary_path}')

    
    info(f'{args.num_runs}-time runs')
    for run_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + run_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

    if len(run_records) > 0:
        if args.minimize_score:
            best_run = min(run_records, key=lambda x: x['avg_test_score'])
        else:
            best_run = max(run_records, key=lambda x: x['avg_test_score'])
        info(f'Best seed by test {args.metric}: {best_run["seed"]} | '
             f'score={best_run["avg_test_score"]:.6f} | model={best_run["model_path"]}')

    
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')


    return mean_score, std_score


if __name__ == '__main__':
    args = parse_train_args()
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    mean_auc_score, std_auc_score = run_stat(args, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')
