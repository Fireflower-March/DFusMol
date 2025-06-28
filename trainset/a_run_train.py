from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List

import numpy as np
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .training import train
from dataset import *
from modelset.model import build_model
from trainset.nn_utils import param_count
from trainset.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from torch.optim.lr_scheduler import ExponentialLR


def a_run_training(args: Namespace,data, logger: Logger = None) -> List[float]:

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
# =============================================================================
#     debug(pformat(vars(args)))
# =============================================================================

    # Get data
    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    args.num_tasks = data.num_tasks()  #任务数
    info(f'Number of tasks = {args.num_tasks}')
    
    # Split data
    debug(f'Splitting data with seed {args.seed}')

    print('='*100)
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)  #每个任务的0和1的比例 [[0.3,0.7],[],[]..]
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:  #对数据集做归一化处理，避免因特征尺度不同而影响模型的训练和预测结果
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

    else:
        scaler = None


    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)


    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))


    save_dir = os.path.join(args.save_dir)
    makedirs(save_dir)

    # Load/build model
    debug(f'Building model')
    model = build_model(args)

    
    debug(model)
    debug(f'Number of parameters = {param_count(model):,}')
    model = model.cuda()


    # Ensure that model is saved in correct location for evaluation if 0 epochs
    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

    # Optimizers
    optimizer = build_optimizer(model, args)

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)

    # Run training
    all_loss_values = []
    all_alpha_values = []

    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch, n_iter = 0, 0
    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')

        n_iter,loss_values= train(
            model=model,
            data=train_data,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
        )

        all_loss_values.append(loss_values)
        all_alpha_values.append(model.encoder_2.alpha.item())

        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        val_scores = evaluate(
            model=model,
            data=val_data,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            batch_size=args.batch_size,
            dataset_type=args.dataset_type,
            scaler=scaler,
            logger=logger
        )
        # Average validation score
        avg_val_score = np.nanmean(val_scores)
        info(f'Validation {args.metric} = {avg_val_score:.6f}')
        
        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
            
        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'test {args.metric} = {avg_test_score:.6f}')
        

        # Save model checkpoint if improved validation score
        if args.minimize_score and avg_val_score < best_score or \
                not args.minimize_score and avg_val_score > best_score:
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args) 


    # Evaluate on test set using model with best validation score
    info(f'Model best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
    model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
    
    test_preds = predict(
        model=model,
        data=test_data,
        batch_size=args.batch_size,
        scaler=scaler
    )
    test_scores = evaluate_predictions(
        preds=test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average test score
    avg_test_score = np.nanmean(test_scores)
    info(f'Model test {args.metric} = {avg_test_score:.6f}')


    return test_scores