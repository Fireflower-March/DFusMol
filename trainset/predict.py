from typing import List

import torch
import torch.nn as nn
from tqdm import trange
from dataset import *
import numpy as np


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:

    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):  #步长为iter_step，这里如果num_iters为205，iter_step为256，那只执行i为0
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch = mol_batch.smiles()
        info_match = collate_fn(mol_batch)
        # Run model
        batch = smiles_batch

        step = 'finetune'
        with torch.no_grad():
            batch_preds = model(step, batch,info_match)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
