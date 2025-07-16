import torch
import numpy as np

def collate_fn(batch):
    # Batch data processing to ensure data extraction from 'molecule_info'
    keys = batch[0].molecule_info.keys()  # Using modified key-value paths
    collated_batch = {}

    for key in keys:
        # Detect whether the data is two-dimensional and process it accordingly
        if batch[0].molecule_info[key].ndim in (2, 0): 
            max_len_1 = max((item.molecule_info[key].shape[0] if item.molecule_info[key].size > 1 else 0)for item in batch)
            max_len_2 = max((item.molecule_info[key].shape[1] if item.molecule_info[key].size > 1 else 0)for item in batch)
            if(key=='sum_atoms'):
                padded = torch.full((len(batch), max_len_1, max_len_2), fill_value=1, dtype=torch.float32)
            else:
                padded = torch.full((len(batch), max_len_1, max_len_2), fill_value=0, dtype=torch.float32)
            for i, item in enumerate(batch):
                data = torch.tensor(item.molecule_info[key], dtype=torch.float32)
                if data.dim() != 0:
                    padded[i, :data.shape[0], :data.shape[1]] = data
        else:
            # Process one-dimensional data
            max_len_1 = max((item.molecule_info[key].shape[0] if item.molecule_info[key].size > 1 else 0)for item in batch)
            padded = torch.full((len(batch), max_len_1), fill_value=0, dtype=torch.long)
            for i, item in enumerate(batch):
                data = torch.tensor(item.molecule_info[key], dtype=torch.long)
                if data.dim() != 0:
                    padded[i, :len(data)] = data

        collated_batch[key] = padded

    return collated_batch


### 51 51