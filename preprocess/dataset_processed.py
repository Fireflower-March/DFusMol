from cProfile import label
import pandas as pd
import numpy as np
import torch
import networkx as nx
from rdkit import Chem
from utils import *
from mol_graph import array_rep_from_smiles

def get_adj_matrix(num_list, edges):
    adjoin_matrix = np.eye(len(num_list))
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return adjoin_matrix

def get_dist_matrix(num_list, edges):
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.full((len(num_list), len(num_list)), 1e9)
    np.fill_diagonal(dist_matrix, 0)
    graph_nodes = sorted(make_graph.nodes)
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for dist in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[dist].items(), key=lambda x: x[0]))
        temp_node_dist_dict = {i: node_relative_distance.get(i, 1e9) for i in graph_nodes}
        temp_node_dist_list = list(temp_node_dist_dict.values())
        dist_matrix[dist][graph_nodes] = temp_node_dist_list
    return dist_matrix.astype(np.float32)

def molgraph_rep(smi, cliques):
    def atom_to_motif_match(atom_order, cliques):
        atom_order = atom_order.tolist()
        temp_matrix = np.zeros((len(cliques), len(atom_order)))
        for th, cli in enumerate(cliques):
            for i in cli:
                temp_matrix[th, atom_order.index(i)] = 1
        return temp_matrix

    def get_adj_dist_matrix(mol_graph, smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        num_atoms = mol.GetNumAtoms()
        adjoin_matrix_temp = np.eye(num_atoms)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = (adjoin_matrix_temp + adj_matrix)[:, mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        dist_matrix = Chem.GetDistanceMatrix(mol)[:, mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        return adj_matrix, dist_matrix

    single_dict = {
        # 'input_atom_features': [],
        'atom_match_matrix': [],
        'sum_atoms': [],
        'adj_matrix': [],
        'dist_matrix': []
    }
    array_rep = array_rep_from_smiles(smi)
    single_dict['atom_match_matrix'] = atom_to_motif_match(array_rep['rdkit_ix'], cliques)
    single_dict['sum_atoms'] = np.reshape(np.sum(single_dict['atom_match_matrix'], axis=1), (-1, 1))
    adj_matrix, dist_matrix = get_adj_dist_matrix(array_rep, smi)
    single_dict['adj_matrix'] = adj_matrix
    single_dict['dist_matrix'] = dist_matrix
    single_dict = {key: np.array(value, dtype='float32') for key, value in single_dict.items()}
    return single_dict

