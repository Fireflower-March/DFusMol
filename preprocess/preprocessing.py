from utils import Mol_Tokenizer
from dataset_processed import get_adj_matrix,get_dist_matrix,molgraph_rep
import pandas as pd
import numpy as np
import tqdm 
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

data = 'toxcast'
def datapro():
    repo_root = Path(__file__).resolve().parent.parent

    toy_dataset = pd.read_csv(repo_root / 'data' / f'{data}.csv')

    unique_SMILES_in_toy_dataset = []
    unique_SMILES_in_toy_dataset.extend(toy_dataset.smiles)

    unique_SMILES_in_toy_dataset = set(unique_SMILES_in_toy_dataset)

    all_drugs_dict = {i:{'adj_matrix':[],'dist_matrix':[],'nums_list':[],'cliques':[],'single_dict':[]} for i in unique_SMILES_in_toy_dataset} 

    ## load tokenizer
    tokenizer = Mol_Tokenizer(repo_root / 'preprocess' / 'token_id.json')

    for drug in tqdm.tqdm(unique_SMILES_in_toy_dataset):
        nums_list1, edges1,cliques1= tokenizer.tokenize(drug)
        dist_matrix1 = get_dist_matrix(nums_list1,edges1)
        adjoin_matrix1 = get_adj_matrix(nums_list1,edges1)
        all_drugs_dict[drug]['adj_matrix']= adjoin_matrix1 
        all_drugs_dict[drug]['dist_matrix']= dist_matrix1 
        all_drugs_dict[drug]['nums_list']= nums_list1
        all_drugs_dict[drug]['cliques'] = cliques1
        all_drugs_dict[drug]['single_dict'] = molgraph_rep(drug,cliques1)
        
    np.save('preprocessed_molecular_'+data+'.npy',all_drugs_dict)

if __name__ == '__main__':
    datapro()
