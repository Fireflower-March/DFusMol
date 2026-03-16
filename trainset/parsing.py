from argparse import ArgumentParser, Namespace
import json
import os
import pickle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_PATH = str(REPO_ROOT / 'preprocess' / 'token_id.json')

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file',
                        default='M_CYP1A2I_I.csv')

    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')

              
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory where model checkpoints will be saved')



    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'multiclass'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.',
                        default='regression') # classification
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes when running multiclass classification')


    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined','cluster_balanced'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs when performing k independent runs')


    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_runs` > 1, the first run uses this seed and all'
                             'subsequent runs add 1 to the seed.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')

    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')

    # Model arguments
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature of contrastive learning')
    parser.add_argument('--encoder_name', type=str, default='CMPNN',
                        choices=['CMPNN', 'MPNN'],
                        help='Name of the encoder')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU'],
                        help='Activation function')                  
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')

    parser.add_argument("--dump_path", default="dumped", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_name", default="", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--exp_id", default="", type=str, required=True,
                        help="Experiment ID")
    
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH, type=str, 
                        help="Experiment ID")



def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    assert args.data_path is not None
    assert args.dataset_type is not None

    args.cuda = torch.cuda.is_available()

    args.features_scaling = True

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'multiclass':
            args.metric = 'cross_entropy'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'mse', 'r2']) or
            (args.dataset_type == 'multiclass' and args.metric in ['cross_entropy', 'accuracy'])):
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    args.minimize_score = args.metric in ['rmse', 'mae', 'mse', 'cross_entropy']
    args.map_dict = 'preprocessed_molecular_'+args.exp_id+'.npy'

    args.num_lrs = 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    if args.test:
        args.epochs = 0


def parse_train_args() -> Namespace:

    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()  

    return args
