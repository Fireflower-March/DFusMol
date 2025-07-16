from cProfile import label
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GraphBertDatasetFineTune(Dataset):
    def __init__(self, dataset, tokenizer, map_dict, label_field='DDI'):
        self.dataset = dataset.reset_index(drop=True)
        self.label_field = label_field
        self.tokenizer = tokenizer
        self.pad_value = self.tokenizer.vocab['<pad>']
        self.map_dict = map_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        smiles1 = self.dataset.loc[idx, 'drug_A']
        label = int(self.dataset.loc[idx, self.label_field])
        data = self.numerical_seq(smiles1, label)
        return data

    def numerical_seq(self, smiles1):
        
        nums_list1 = self.map_dict[smiles1]['nums_list'] 
        dist_matrix1 = self.map_dict[smiles1]['dist_matrix'] 
        adjoin_matrix1 = self.map_dict[smiles1]['adj_matrix'] 
        single_dict1_atom = self.map_dict[smiles1]['single_dict'] 

        
        nums_list1 = [self.tokenizer.vocab['<global>']] + nums_list1

        temp1 = np.ones((len(nums_list1), len(nums_list1)))
        temp1[1:, 1:] = adjoin_matrix1 
        adjoin_matrix1 = (1 - temp1) * (-1e9)

        temp1_dist = np.ones((len(nums_list1), len(nums_list1))) 
        temp1_dist[0][0] = 0 
        temp1_dist[1:, 1:] = dist_matrix1
        dist_matrix1 = temp1_dist 
        

        
        atom_features1 = single_dict1_atom['input_atom_features']
        dist_matrix1_atom = single_dict1_atom['dist_matrix']
        adjoin_matrix1_atom = single_dict1_atom['adj_matrix']
        adjoin_matrix1_atom = (1 - adjoin_matrix1_atom) * (-1e9)  #
        atom_match_matrix1 = single_dict1_atom['atom_match_matrix']
        sum_atoms1 = single_dict1_atom['sum_atoms']
        

        
        x1 = np.array(nums_list1).astype('int64')

        return {
            'molecule_sequence1': x1,
            'adj_matrix1': adjoin_matrix1,
            'dist_matrix1': dist_matrix1,
            'atom_features1': atom_features1,
            'adjoin_matrix1_atom': adjoin_matrix1_atom,
            'dist_matrix1_atom': dist_matrix1_atom,
            'atom_match_matrix1': atom_match_matrix1,
            'sum_atoms1': sum_atoms1,
        }

def collate_fn(batch):
    
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if key == 'label':
            collated_batch[key] = torch.tensor([item[key][0] for item in batch], dtype=torch.long)
        else:
            
            max_len_1 = max([item[key].shape[0] for item in batch])
            if len(batch[0][key].shape) == 2:
                max_len_2 = max([item[key].shape[1] for item in batch])
                padded = torch.full((len(batch), max_len_1, max_len_2), fill_value=0, dtype=torch.float32)
                for i, item in enumerate(batch):
                    data = torch.tensor(item[key], dtype=torch.float32)
                    padded[i, :data.shape[0], :data.shape[1]] = data
            else:
                padded = torch.full((len(batch), max_len_1), fill_value=0, dtype=torch.long)
                for i, item in enumerate(batch):
                    data = torch.tensor(item[key], dtype=torch.long)
                    padded[i, :data.shape[0]] = data
            collated_batch[key] = padded
    return collated_batch
