import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Mol_Tokenizer
from modelset import *
from argparse import ArgumentParser, Namespace

class MoleculePropertyPredictionModel(nn.Module):
    def __init__(self, args):
        super(MoleculePropertyPredictionModel, self).__init__()

        d_model = 256
        num_heads = 4  
        num_layers = 3 
        dff = d_model * 2 
        dropout_rate = args.dropout
        self.args = args
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        # atom encoder
        self.atom_encoder = EncoderModel_atom(
            num_layers=num_layers,
            d_model=d_model,   
            num_heads=num_heads, 
            dff=dff, 
            rate=dropout_rate
        )

        # motif encoder
        self.motif_encoder = EncoderModel_motif(
            num_layers=num_layers,  
            input_vocab_size= Mol_Tokenizer(args.tokenizer).get_vocab_size,
            d_model=d_model,  
            num_heads=num_heads,  
            dff=dff,    
            rate=dropout_rate   
        )

        
        self.fc1 = nn.Linear(d_model, d_model) 
        self.dropout1 = nn.Dropout(dropout_rate)


        
        self.Wa = nn.Linear(d_model, d_model)

    def forward(self,mol_vecs_padded,mol_vecs,info):
        ## [batch_size, atom_len, f_dim]  {'...':...,'...':...,}
        atom_inputs = mol_vecs_padded
        atom_adj_inputs= info['adj_matrix_atom']
        atom_dist_inputs= info['adj_matrix_atom']
        atom_match_matrix= info['atom_match_matrix']
        sum_atoms= info['sum_atoms']
        motif_inputs= info['molecule_sequence']
        motif_adj_inputs=info['adj_matrix']
        motif_dist_inputs = info['dist_matrix']
        if self.args.cuda or next(self.parameters()).is_cuda:
            atom_adj_inputs, atom_dist_inputs, atom_match_matrix, sum_atoms, motif_inputs,motif_adj_inputs,motif_dist_inputs = (
                    atom_adj_inputs.cuda(), atom_dist_inputs.cuda(), atom_match_matrix.cuda(), \
                    sum_atoms.cuda(), motif_inputs.cuda(),motif_adj_inputs.cuda(),motif_dist_inputs.cuda())
        
        Outseq_atom, _, _, _ = self.atom_encoder(
            atom_inputs,
            adjoin_matrix=atom_adj_inputs,
            dist_matrix=atom_dist_inputs,
            atom_match_matrix=atom_match_matrix,
            sum_atoms=sum_atoms
        )

        
        molecule_trans, _, _, _ = self.motif_encoder(
            motif_inputs,
            atom_level_features=Outseq_atom,
            adjoin_matrix=motif_adj_inputs,
            dist_matrix=motif_dist_inputs
        )

        
        molecule_trans = molecule_trans[:, 0, :]



        
        output = self.fc1(molecule_trans)
        output = F.relu(output)
        output = self.dropout1(output)


         
        epsilon = 1e-5
        a_sums = mol_vecs.sum(dim=1) 
        b_sums = output.sum(dim=1) 

          
        scaling_factors = (a_sums + epsilon) / (b_sums + epsilon)
        scaling_factors = scaling_factors.unsqueeze(1) 
        output = output * scaling_factors

        final_output = mol_vecs + self.alpha * output

        return final_output
