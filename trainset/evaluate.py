import logging
from typing import Callable, List

import torch.nn as nn

from .predict import predict
from dataset import *


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         dataset_type: str,
                         logger: logging.Logger = None) -> List[float]:

    info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        if dataset_type == 'multiclass':
            results.append(metric_func(valid_targets[i], valid_preds[i], labels=list(range(len(valid_preds[i][0])))))
        else:
            results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:

    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets()

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results
