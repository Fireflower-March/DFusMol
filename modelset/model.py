from argparse import Namespace

from .cmpn import CMPN
from .lgam import MoleculePropertyPredictionModel
from trainset.nn_utils import get_activation_function, initialize_weights
import pdb
import logging
from mimetypes import init
from turtle import forward, hideturtle, up
import torch
import torch.nn as nn
import torch.nn.functional as F



class MoleculeModel(nn.Module):

    def __init__(self, classification: bool, multiclass: bool):

        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):

        self.encoder_1 = CMPN(args)
        self.encoder_2 = MoleculePropertyPredictionModel(args)

    def create_ffn(self, args: Namespace):

        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        first_linear_dim = args.hidden_size


        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:  #2
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self,step,smiles,info):

        mol_vecs_padded,mol_vecs = self.encoder_1(step,smiles)    #原子级进行内部聚合  # [batch_size, max_atom_len, dim]
        info_output = self.encoder_2(mol_vecs_padded,mol_vecs,info)    #与基序一同进入注意力
        output = self.ffn(info_output)
        # output = self.ffn(mol_vecs)
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output



def build_model(args: Namespace) -> nn.Module:

    output_size = args.num_tasks  ############这里设置了输出的任务数
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)
    return model
