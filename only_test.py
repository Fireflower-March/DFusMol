import argparse
from pathlib import Path
import json
import numpy as np
import torch
import warnings
from rdkit import RDLogger

from dataset import get_data, split_data
from trainset.evaluate import evaluate
from trainset.utils import get_metric_func, load_checkpoint, load_scalers, set_seed

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_PATH = REPO_ROOT / 'preprocess' / 'token_id.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit-only evaluation: split data then evaluate a provided checkpoint on test set.'
    )
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file.')
    parser.add_argument('--dataset_type', type=str, default='classification',
                        choices=['classification', 'regression', 'multiclass'],
                        help='Dataset type to select metric behavior.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric for evaluation. If not set, inferred from dataset_type.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint (*.pt) to evaluate.')
    parser.add_argument('--tokenizer', type=str, default=str(DEFAULT_TOKENIZER_PATH),
                        help='Path to tokenizer json file.')
    parser.add_argument('--map_dict', type=str, default=None,
                        help='Path to preprocessed molecular npy map. Defaults to preprocessed_molecular_<dataset>.npy.')
    parser.add_argument('--split_type', type=str, default='scaffold_balanced',
                        choices=['random', 'scaffold_balanced', 'cluster_balanced'],
                        help='Data split strategy.')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Train/val/test split sizes, e.g. 0.8 0.1 0.1.')
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed used for split (must match the seed used during training).')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation.')
    parser.add_argument('--gpu', type=int, default=None, help='CUDA device id. If omitted, uses CPU.')
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes if dataset_type=multiclass.')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Optional JSON output path for test results.')
    return parser.parse_args()


def resolve_metric(args: argparse.Namespace) -> str:
    if args.metric is not None:
        return args.metric
    if args.dataset_type == 'classification':
        return 'auc'
    if args.dataset_type == 'multiclass':
        return 'cross_entropy'
    return 'rmse'


def resolve_map_dict(args: argparse.Namespace) -> str:
    if args.map_dict is not None:
        return args.map_dict
    dataset_name = Path(args.data_path).stem
    return f'preprocessed_molecular_{dataset_name}.npy'


def main():
    args = parse_args()
    args.metric = resolve_metric(args)
    args.map_dict = resolve_map_dict(args)
    args.cuda = torch.cuda.is_available() and args.gpu is not None

    if args.cuda:
        torch.cuda.set_device(args.gpu)

    set_seed(args.seed)

    data = get_data(path=args.data_path, args=args, logger=None)
    _, _, test_data = split_data(
        data=data,
        split_type=args.split_type,
        sizes=tuple(args.split_sizes),
        seed=args.seed,
        args=args,
        logger=None
    )

    model = load_checkpoint(args.checkpoint_path, cuda=args.cuda, logger=None)
    data_scaler, _ = load_scalers(args.checkpoint_path)
    scaler = data_scaler if args.dataset_type == 'regression' else None

    metric_func = get_metric_func(args.metric)
    test_scores = evaluate(
        model=model,
        data=test_data,
        num_tasks=test_data.num_tasks(),
        metric_func=metric_func,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        scaler=scaler,
        logger=None
    )
    avg_test_score = float(np.nanmean(test_scores))

    result = {
        'data_path': args.data_path,
        'checkpoint_path': args.checkpoint_path,
        'seed': args.seed,
        'split_type': args.split_type,
        'split_sizes': list(args.split_sizes),
        'metric': args.metric,
        'avg_test_score': avg_test_score,
        'task_scores': [float(x) for x in test_scores]
    }

    print(json.dumps(result, indent=2))

    if args.result_path is not None:
        with open(args.result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f'Saved result json: {args.result_path}')


if __name__ == '__main__':
    main()
