import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Mol_Tokenizer
from modelset import *
from argparse import ArgumentParser, Namespace

# param = {'name': 'Small', 'num_layers': 4, 'num_heads': 8, 'd_model': 512}
class MoleculePropertyPredictionModel(nn.Module):
    def __init__(self, args):
        super(MoleculePropertyPredictionModel, self).__init__()
        # d_model = args.arch['d_model']  #512
        # num_heads = args.arch['num_heads']  #8
        # num_layers = args.arch['num_layers']  #4
        # dff = d_model #512
        d_model = 256
        num_heads = 4  #over
        num_layers = 3 #over
        dff = d_model * 2 #512
        dropout_rate = args.dropout
        self.args = args
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        # atom encoder
        self.atom_encoder = EncoderModel_atom(
            num_layers=num_layers,
            d_model=d_model,   # 256
            num_heads=num_heads, #8
            dff=dff, #512
            rate=dropout_rate #0.1
        )

        # motif encoder
        self.motif_encoder = EncoderModel_motif(
            num_layers=num_layers,  #4
            input_vocab_size= Mol_Tokenizer(args.tokenizer).get_vocab_size,
            d_model=d_model,  #256
            num_heads=num_heads,  #8
            dff=dff,    #512
            rate=dropout_rate   #0.1
        )

        
        self.fc1 = nn.Linear(d_model, d_model) #512-256
        self.dropout1 = nn.Dropout(dropout_rate)


        # 为了避免高相似度分数的线性层
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
        # 分子的原子级编码
        Outseq_atom, _, _, _ = self.atom_encoder(
            atom_inputs,
            adjoin_matrix=atom_adj_inputs,
            dist_matrix=atom_dist_inputs,
            atom_match_matrix=atom_match_matrix,
            sum_atoms=sum_atoms
        )

        # 分子的基元级编码
        molecule_trans, _, _, _ = self.motif_encoder(
            motif_inputs,
            atom_level_features=Outseq_atom,
            adjoin_matrix=motif_adj_inputs,
            dist_matrix=motif_dist_inputs
        )

        # # 取出 [CLS] 位置的向量，并通过线性层
        molecule_trans = molecule_trans[:, 0, :]



        # 通过分类层进行预测
        output = self.fc1(molecule_trans)
        output = F.relu(output)
        output = self.dropout1(output)


        # 定义平衡因子 ε，避免异常值造成缩放比例极端
        epsilon = 1e-5
        a_sums = mol_vecs.sum(dim=1) 
        b_sums = output.sum(dim=1) 

        # 计算缩放比例，并避免 b_sums 中的异常小值导致比例过大
        scaling_factors = (a_sums + epsilon) / (b_sums + epsilon)
        scaling_factors = scaling_factors.unsqueeze(1) 
        output = output * scaling_factors

        final_output = mol_vecs + self.alpha * output

        return final_output
