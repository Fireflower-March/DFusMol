import torch
import torch.nn as nn
import torch.nn.functional as F

def rescale_distance_matrix(w):  # For global
    constant_value = torch.tensor(1.0)
    return (constant_value + torch.exp(constant_value)) / (constant_value + torch.exp(constant_value - w))

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def create_padding_mask_atom(batch_data):
    padding_mask = (batch_data.sum(dim=-1) == 0).float()
    return padding_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]



def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    """Calculate the attention weights."""
    if dist_matrix is not None:
        matmul_qk = F.relu(torch.matmul(q, k.transpose(-2, -1))) #(batch_size, num_heads, atom_len, atom_len)
        dist_matrix = rescale_distance_matrix(dist_matrix) 
        dk = k.size(-1)
        scaled_attention_logits = (matmul_qk * dist_matrix) / torch.sqrt(torch.tensor(dk, dtype=q.dtype))
    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=q.dtype))
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights #[batch_size, num_heads, atom_len, depth_v], [batch_size, num_heads, atom_len, atom_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, atom_len, depth)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, atom_len, 256)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, atom_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        #[batch_size, num_heads, atom_len, depth_v], [batch_size, num_heads, atom_len, atom_len]
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, atom_len, num_heads, depth)

        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)  # (batch_size, atom_len, d_model)

        output = self.dense(concat_attention)  # (batch_size, atom_len, d_model)

        return output, attention_weights  #[batch_size, atom_len, d_model], [batch_size, num_heads, atom_len, atom_len]

def feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.GELU(),
        nn.Linear(dff, d_model)
    )

class EncoderLayer(nn.Module):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff, rate):  #256.512
        super(EncoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.split(x, x.size(-1) // 2, dim=-1) #[batch_size,atom_len,256]

        
        x_l, attention_weights_local = self.mha1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None)
        x_g, attention_weights_global = self.mha2(
            x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix)
        

        #x_l,x_g:[batch_size, atom_len, 256],
        #attention_weights_local,attention_weights_global:[batch_size, num_heads, atom_len, atom_len]

        attn_output = torch.cat([x_l, x_g], dim=-1) #[batch_size, atom_len, 512]
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output) 

        ffn_output = self.ffn(out1)  #MLP
        ffn_output = self.dropout2(ffn_output) 
        out2 = self.layer_norm2(out1 + ffn_output) 
        x_l_g = out2
        return x_l_g, attention_weights_local, attention_weights_global

class EncoderModel_atom(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):  #256,512
        super(EncoderModel_atom, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.global_embedding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(rate)
        # self.encoder_layers = nn.ModuleList([
        #     EncoderLayer(int(d_model), num_heads, dff, rate)
        #     for _ in range(self.num_layers)
        # ])

    def forward(self, x, adjoin_matrix=None,
                dist_matrix=None, atom_match_matrix=None, sum_atoms=None):
        batch_size = x.size(0) # x=[batch_size,atom_len,f_dim]
        encoder_padding_mask = create_padding_mask_atom(x) # [batch_size, 1, 1, atom_len] 
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        # x = self.embedding(x)
        # x = self.dropout(x)

        attention_weights_list_local = []
        attention_weights_list_global = []
        # for i in range(self.num_layers):
        #     x, attention_weights_local, attention_weights_global = self.encoder_layers[i](
        #         x, encoder_padding_mask, adjoin_matrix, dist_matrix=dist_matrix)
        #     attention_weights_list_local.append(attention_weights_local)
        #     attention_weights_list_global.append(attention_weights_global)

        # x :[batch_size, max_atom_len, 256]
        # atom_match_matrix :[batch_size, num_motifs, num_atoms_in_batch(atom_len)]

        
        if atom_match_matrix is not None and sum_atoms is not None:
            x = torch.matmul(atom_match_matrix, x)
            x = x / sum_atoms
            x = self.global_embedding(x)
        else:
            
            pass

        return x, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask
