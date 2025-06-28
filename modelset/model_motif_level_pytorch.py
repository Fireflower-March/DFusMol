import torch
import torch.nn as nn
import torch.nn.functional as F

def rescale_distance_matrix(w):  # For global
    constant_value = torch.tensor(1.0)
    return (constant_value + torch.exp(constant_value)) / (constant_value + torch.exp(constant_value - w))

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def create_padding_mask(batch_data):
    padding_mask = (batch_data == 0).float()
    return padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    """Calculate the attention weights."""
    dist_matrix = None
    if dist_matrix is not None:
        matmul_qk = F.relu(torch.matmul(q, k.transpose(-2, -1)))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        dk = k.size(-1)
        scaled_attention_logits = (matmul_qk * dist_matrix) / torch.sqrt(torch.tensor(dk, dtype=q.dtype))
    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=q.dtype))
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # adjoin_matrix = None
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

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
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

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
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.split(x, x.size(-1) // 2, dim=-1)
        x_l, attention_weights_local = self.mha1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None)
        x_g, attention_weights_global = self.mha2(
            x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix)
        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)  #残差连接+归一化 Add & Norm

        ffn_output = self.ffn(out1)     # feedforward
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)  #第二次残差连接+归一化 Add & Norm
        x_l_g = out2
        return x_l_g, attention_weights_local, attention_weights_global

class EncoderModel_motif(nn.Module):
    def __init__(self, num_layers, input_vocab_size,
                 d_model, num_heads, dff, rate=0.1):
        super(EncoderModel_motif, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, self.d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)
        ])

    def forward(self, x, atom_level_features, adjoin_matrix=None, dist_matrix=None):
        encoder_padding_mask = create_padding_mask(x)
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)
        x = self.embedding(x) #[batch_size,max_motif_len,256]
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.dropout(x)
        x_temp = x[:, 1:, :] + atom_level_features #排除掉每个分子的第一个全局基序不修改
        x = torch.cat([x[:, 0:1, :], x_temp], dim=1) #重新添加上全局基序 [batch_size,max_motif_len,256]
        temp = x
        attention_weights_list_local = []
        attention_weights_list_global = []
        for i in range(self.num_layers):
            x, attention_weights_local, attention_weights_global = self.encoder_layers[i](
                x, encoder_padding_mask, adjoin_matrix, dist_matrix=dist_matrix)
            attention_weights_list_local.append(attention_weights_local)
            attention_weights_list_global.append(attention_weights_global)
        # x = temp + x
        return x, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask
