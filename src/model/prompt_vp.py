import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_head, n_feat, o_feat, dropout_rate, q_in_dim, k_in_dim, v_in_dim):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(q_in_dim, n_feat)
        self.linear_k = nn.Linear(k_in_dim, n_feat)
        self.linear_v = nn.Linear(v_in_dim, n_feat)
        self.linear_out = nn.Linear(n_feat, o_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward(self, query, key, value):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        n_batch = value.size(0)
        attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k))
        return self.linear_out(x)


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self,
                 idim,
                 hidden_units,
                 dropout_rate,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class PromptVPEncoderLayer(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, k_in_dim, v_in_dim):
        super(PromptVPEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            n_head=n_head, n_feat=n_feat, dropout_rate=0.,
            q_in_dim=q_in_dim, k_in_dim=q_in_dim, v_in_dim=q_in_dim)
        self.src_attn = MultiHeadedAttention(
            n_head=n_head, n_feat=n_feat, dropout_rate=0.,
            q_in_dim=q_in_dim, k_in_dim=k_in_dim, v_in_dim=v_in_dim)
        self.norm1 = nn.LayerNorm(n_feat, eps=1e-5)
        self.norm2 = nn.LayerNorm(n_feat, eps=1e-5)
        self.norm3 = nn.LayerNorm(n_feat, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = PositionwiseFeedForward(n_feat, n_feat, dropout_rate)

    def forward(self, q, k, v):
        residual = q
        x = residual + self.dropout(self.self_attn(q, q, q))
        x = self.norm1(x)
        residual = x
        x = residual + self.dropout(self.src_attn(x, k, v))
        x = self.norm2(x)
        residual = x
        x = residual + self.dropout(self.ffn(x))
        x = self.norm3(x)
        return x, k, v


class PromptEncoderLayer(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate, q_in_dim):
        super(PromptEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            n_head=n_head, n_feat=n_feat, dropout_rate=0.,
            q_in_dim=q_in_dim, k_in_dim=q_in_dim, v_in_dim=q_in_dim)
        self.norm1 = nn.LayerNorm(n_feat, eps=1e-5)
        self.norm2 = nn.LayerNorm(n_feat, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = PositionwiseFeedForward(n_feat, n_feat, dropout_rate)

    def forward(self, q, k, v):
        residual = q
        x = residual + self.dropout(self.self_attn(q, q, q))
        x = self.norm1(x)
        residual = x
        x = residual + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x, k, v


class PromptVPEncoder(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, k_in_dim, v_in_dim, num_blocks):
        super(PromptVPEncoder, self).__init__()
        self.prompt_vp_encoders = torch.nn.ModuleList([
            PromptVPEncoderLayer(n_head, n_feat, dropout_rate,
                                 q_in_dim, k_in_dim, v_in_dim)
            for _ in range(num_blocks)
        ])
        self.mask_prob = 0.08

    def forward(self, feature):
        query = feature['prompt'].transpose(1, 2)
        key = feature['vp'].unsqueeze(1).repeat(1, query.shape[1], 1)
        value = feature['vp'].unsqueeze(1).repeat(1, query.shape[1], 1)
        if self.training:
            # probability_matrix = torch.full([query.shape[0], query.shape[1], 1], self.mask_prob)
            probability_matrix = torch.full(query.shape, self.mask_prob)
            mask = torch.bernoulli(probability_matrix).bool().to(query.device)
            query *= ~mask
        for layer in self.prompt_vp_encoders:
            query, key, value = layer(query, key, value)
        return query


class PromptEncoder(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, num_blocks):
        super(PromptEncoder, self).__init__()
        self.prompt_vp_encoders = torch.nn.ModuleList([
            PromptEncoderLayer(n_head, n_feat, dropout_rate, q_in_dim)
            for _ in range(num_blocks)
        ])
        self.mask_prob = 0.08

    def forward(self, feature):
        query = feature['prompt'].transpose(1, 2)
        key = feature['vp'].unsqueeze(1).repeat(1, query.shape[1], 1)
        value = feature['vp'].unsqueeze(1).repeat(1, query.shape[1], 1)
        if self.training:
            probability_matrix = torch.full([query.shape[0], query.shape[1], 1], self.mask_prob)
            # probability_matrix = torch.full(query.shape, self.mask_prob)
            mask = torch.bernoulli(probability_matrix).bool().to(query.device)
            query *= ~mask
        for layer in self.prompt_vp_encoders:
            query, key, value = layer(query, key, value)
        return query


class MRTELayer(nn.Module):
    def __init__(self, n_head, n_feat, o_feat, dropout_rate, q_in_dim, k_in_dim, v_in_dim):
        super(MRTELayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            n_head=n_head, n_feat=n_feat,
            o_feat=o_feat,
            dropout_rate=0.,
            q_in_dim=q_in_dim, k_in_dim=k_in_dim, v_in_dim=v_in_dim)
        self.norm1 = nn.LayerNorm(n_feat, eps=1e-5)
        self.norm2 = nn.LayerNorm(n_feat, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = PositionwiseFeedForward(n_feat, n_feat, dropout_rate)

    def forward(self, q, k, v):
        residual = q
        x = residual + self.dropout(self.self_attn(q, k, v))
        x = self.norm1(x)
        residual = x
        x = residual + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x, k, v


class MRTE(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, k_in_dim, v_in_dim, num_blocks):
        super(MRTE, self).__init__()
        self.prompt_vp_encoders = torch.nn.ModuleList([
            MRTELayer(n_head,
                      n_feat=n_feat,
                      o_feat=n_feat,
                      dropout_rate=dropout_rate,
                      q_in_dim=q_in_dim,
                      k_in_dim=k_in_dim+80,
                      v_in_dim=v_in_dim)
            for _ in range(num_blocks)
        ])
        self.vp_proj = nn.Linear(256, 80, bias=False)

    def forward(self, cond, prompts, spks):
        query = cond   # B,T1,D1
        key = prompts  # B,T2,D2
        value = prompts
        GE = self.vp_proj(spks).unsqueeze(1).repeat(1, key.shape[1], 1)
        key = torch.cat((key, GE), dim=-1)
        for idx, layer in enumerate(self.prompt_vp_encoders):
            query, key, value = layer(query, key, value)
            # query += GE
        return query

# class MRTE(nn.Module):
#     def __init__(self, n_head, n_feat, dropout_rate,
#                  q_in_dim, k_in_dim, v_in_dim, num_blocks):
#         super(MRTE, self).__init__()
#         self.prompt_vp_encoders = torch.nn.ModuleList([
#             MRTELayer(n_head,
#                       n_feat=n_feat,
#                       o_feat=n_feat,
#                       dropout_rate=dropout_rate,
#                       q_in_dim=q_in_dim,
#                       k_in_dim=k_in_dim,
#                       v_in_dim=v_in_dim)
#             for _ in range(num_blocks)
#         ])
#         self.vp_proj = nn.Linear(192, n_feat, bias=False)

#     def forward(self, feature):
#         query = feature['token_emb'].transpose(1, 2)
#         key = feature['prompt'].transpose(1, 2)
#         value = feature['prompt'].transpose(1, 2)
#         GE = self.vp_proj(feature['vp']).unsqueeze(1).repeat(1, query.shape[1], 1)
#         for idx, layer in enumerate(self.prompt_vp_encoders):
#             query, key, value = layer(query, key, value)
#             query += GE
#         return query


class CrossAttentionEncoder(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, k_in_dim, v_in_dim, num_blocks):
        super(CrossAttentionEncoder, self).__init__()
        self.emb = nn.Embedding(8192, 1024)
        self.encoders = torch.nn.ModuleList([
            MRTELayer(n_head,
                      n_feat=n_feat,
                      o_feat=n_feat,
                      dropout_rate=dropout_rate,
                      q_in_dim=q_in_dim,
                      k_in_dim=k_in_dim,
                      v_in_dim=v_in_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, feature):
        query = self.emb(feature['query'])
        key = feature['key'].transpose(1, 2)
        value = feature['value'].transpose(1, 2)
        for idx, layer in enumerate(self.encoders):
            query, key, value = layer(query, key, value)
        return query


class TextEncoder(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, k_in_dim, v_in_dim, num_blocks):
        super(TextEncoder, self).__init__()
        self.text_encoders = torch.nn.ModuleList([
            MRTELayer(n_head,
                      n_feat=n_feat,
                      o_feat=n_feat,
                      dropout_rate=dropout_rate,
                      q_in_dim=q_in_dim,
                      k_in_dim=k_in_dim,
                      v_in_dim=v_in_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, feature):
        query = feature['token_emb'].transpose(1, 2)
        key = feature['bert_hidden']
        value = feature['bert_hidden']
        for idx, layer in enumerate(self.text_encoders):
            query, key, value = layer(query, key, value)
        return query


class TransformerEncoder(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate,
                 q_in_dim, k_in_dim, v_in_dim, num_blocks):
        super(TransformerEncoder, self).__init__()
        self.encoders = torch.nn.ModuleList([
            MRTELayer(n_head,
                      n_feat=n_feat,
                      o_feat=n_feat,
                      dropout_rate=dropout_rate,
                      q_in_dim=q_in_dim,
                      k_in_dim=k_in_dim,
                      v_in_dim=v_in_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for idx, layer in enumerate(self.encoders):
            x, _, _ = layer(x, x, x)
        return x


if __name__ == '__main__':
    # feature = {
    #     'token_emb': torch.rand(2, 1024, 128),
    #     'prompt': torch.rand(2, 1024, 150),
    #     'vp': torch.rand(2, 192)
    # }
    # x = torch.rand(2, 1024, 128)
    # attention = MultiHeadedAttention(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, k_in_dim=192, v_in_dim=192)
    # query = feature['prompt']
    # key = feature['vp'].unsqueeze(1).repeat(1, feature['prompt'].shape[1], 1)
    # value = feature['vp'].unsqueeze(1).repeat(1, feature['prompt'].shape[1], 1)
    # o = attention(query, key, value)
    # encoder_layer = PromptVPEncoderLayer(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, k_in_dim=192,
    #                                      v_in_dim=192)
    # encoder_layer(query, key, value)
    # encoder = PromptVPEncoder(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, k_in_dim=192, v_in_dim=192, num_blocks=2)
    # encoder = PromptEncoder(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, num_blocks=2)
    # encoder = MRTE(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, k_in_dim=1024, v_in_dim=1024, num_blocks=2)
    # encoder = TransformerEncoder(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, k_in_dim=1024, v_in_dim=1024,
    #                              num_blocks=2)
    feature = {
        'token_emb': torch.rand(2, 1024, 128),
        'bert_hidden': torch.rand(2, 64, 768),
    }
    encoder = TextEncoder(n_head=4, n_feat=1024, dropout_rate=0., q_in_dim=1024, k_in_dim=768, v_in_dim=768, num_blocks=2)
    encoder.eval()
    hidden = encoder(feature)
    exit(0)
