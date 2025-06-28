from argparse import Namespace
import csv
from logging import Logger
import pickle
import random
from typing import List, Set, Tuple
import os


from rdkit import Chem
import numpy as np
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split, cluster_split

##
def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def get_smiles(path: str, header: bool = True) -> List[str]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line is a header).

    :param path: Path to a CSV file.
    :param header: Whether the CSV file contains a header (that will be skipped).
    :return: A list of smiles strings.
    """
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)  # Skip header
        smiles = [line[0] for line in reader]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0])

##
def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             logger: Logger = None) -> MoleculeDataset:

    debug = logger.debug if logger is not None else print
    skip_smiles = set()
    count = 0
    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # 跳过第一行

        lines = []
        for line in reader:     #对每一行
            smiles = line[0]

            if smiles in skip_smiles:
                continue
            count += 1
            lines.append(line)
            # if count > 300:
            #     break

        data = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
            tokenizer=args.tokenizer,  # 传入必要的tokenizer
            map_dict=args.map_dict,    # 传入必要的map_dict
            args=args
        ) for i, line in tqdm(enumerate(lines), total=len(lines))
    ])


    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([MoleculeDatapoint([smile]) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data

##
def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:

    assert len(sizes) == 3 and sum(sizes) == 1  
    
    if split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, 
                              logger=logger)

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'cluster_balanced':
        return cluster_split(data, sizes=sizes, balanced=True, seed=seed, 
                              logger=logger)
        
    else:
        raise ValueError(f'split_type "{split_type}" not supported.')

##
def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])  #valid_targets[task_num]=[1.0, 1.0, 1.0,...]

    class_sizes = []
    for task_targets in valid_targets:
        
        assert set(np.unique(task_targets)) <= {0, 1} #确认是二分类

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets) #查看单个任务的标签为1的比例
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes  #返回的每个任务的0和1的比例 [[0.3,0.7],[],[]..]


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors

