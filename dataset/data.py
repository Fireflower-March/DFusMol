from argparse import Namespace
import random
from typing import Callable, List, Union
import json
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
from torch.utils.data.dataset import Dataset
from typing import List, Optional
import numpy as np
from rdkit import Chem

class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 line: List[str],
                 tokenizer=None,
                 map_dict=None,
                 args: Namespace = None):
        if args is not None:
            self.args = args

        self.smiles = line[0]  # SMILES string
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.targets = [float(x) if x != '' else None for x in line[1:]]




        tokenizer = Mol_Tokenizer(args.tokenizer)
        map_dict = np.load(args.map_dict, allow_pickle=True).item()

        # Preprocess and store molecular data
        self.molecule_info = self.preprocess_molecule(self.smiles, tokenizer, map_dict)

    def preprocess_molecule(self, smiles, tokenizer, map_dict):
        """ Preprocess molecular data based on the given SMILES string and tokenizer."""
        # Getting the preprocessing content from map_dict
        molecule_data = map_dict[smiles]

        nums_list = [tokenizer.vocab['<global>']] + molecule_data['nums_list']
        temp_adj = np.ones((len(nums_list), len(nums_list)))
        temp_adj[1:, 1:] = molecule_data['adj_matrix']
        adj_matrix = (1 - temp_adj) * (-1e9)

        temp_dist = np.ones((len(nums_list), len(nums_list)))
        temp_dist[0][0] = 0
        temp_dist[1:, 1:] = molecule_data['dist_matrix']
        dist_matrix = temp_dist

        dist_matrix_atom = molecule_data['single_dict']['dist_matrix']
        temp_adj_atom = np.ones_like(dist_matrix_atom)
        temp_adj_atom = molecule_data['single_dict']['adj_matrix']
        adj_matrix_atom = (1 - temp_adj_atom) * (-1e9)
        atom_match_matrix = molecule_data['single_dict']['atom_match_matrix']
        sum_atoms = molecule_data['single_dict']['sum_atoms']

        return {
            'molecule_sequence': np.array(nums_list).astype('int64'),
            'adj_matrix': adj_matrix,
            'dist_matrix': dist_matrix,
            'adj_matrix_atom': adj_matrix_atom,
            'dist_matrix_atom': dist_matrix_atom,
            'atom_match_matrix': atom_match_matrix,
            'sum_atoms': sum_atoms
        }

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets


##整个数据集的容器
class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def compound_names(self) -> List[str]:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]
    

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]


class Mol_Tokenizer():
    def __init__(self,tokens_id_file):
        self.vocab = json.load(open(r'{}'.format(tokens_id_file),'r'))
        self.MST_MAX_WEIGHT = 100
        self.get_vocab_size = len(self.vocab.keys())
        self.id_to_token = {value:key for key,value in self.vocab.items()}
    def tokenize(self,smiles):
        mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        ids,edge = self.tree_decomp(mol) 
        motif_list = []
        for id_ in ids:
            _,token_mols = self.get_clique_mol(mol,id_)
            token_id = self.vocab.get(token_mols)
            if token_id!=None:
                motif_list.append(token_id)
            else: 
                motif_list.append(self.vocab.get('<unk>'))
        return motif_list,edge,ids
    def sanitize(self,mol):
        try:
            smiles = self.get_smiles(mol)
            mol = self.get_mol(smiles)
        except Exception as e:
            return None
        return mol
    def get_mol(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol
    def get_smiles(self,mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    def get_clique_mol(self,mol,atoms_ids):
    # get the fragment of clique
        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False) 
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = self.copy_edit_mol(new_mol).GetMol()
        new_mol = self.sanitize(new_mol)  # We assume this is not None
        return new_mol,smiles
    def copy_atom(self,atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom
    def copy_edit_mol(self,mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = self.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol
    def tree_decomp(self,mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        # get rings
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Merge Rings with intersection > 2 atoms
        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        # Build edges and add singleton cliques
        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(
                    cnei) > 2):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:  # Multiple (n>2) complex rings
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

        edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # Compute Maximum Spanning Tree
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
        return (cliques, edges)


def hash_to_float(hash_value, range_min=0, range_max=1):
    """
    将哈希值转换为浮点数并映射到指定范围。
    :param hash_value: 哈希值（字符串形式）。
    :param range_min: 映射的最小值。
    :param range_max: 映射的最大值。
    :return: 映射后的浮点数。
    """
    # 将哈希值转换为整数
    int_value = int(hash_value, 16)  # 转换为十进制整数
    # 映射到 [0, 1]
    normalized_value = int_value % (10 ** 8) / (10 ** 8)  # 截断以避免过大数值
    # 映射到指定范围
    return range_min + (range_max - range_min) * normalized_value