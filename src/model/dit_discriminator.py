"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from x_transformers.x_transformers import RotaryEmbedding

from src.model.prompt_vp import MRTE
from src.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    ChunkDiTBlock,
    AdaLayerNorm_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


class AttentionAggregation(nn.Module):
    """
    Aggregates a sequence of features into a single feature vector using cross-attention.
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        # The learnable query token that will gather information from the sequence.
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): The key/value sequence. Shape: (b, n, d)
        Returns:
            torch.Tensor: The aggregated global feature vector. Shape: (b, d)
        """
        b = features.shape[0]
        query = self.query.repeat(b, 1, 1) # Shape: (b, 1, d)
        aggregated_feature, _ = self.attention(query=query, key=features, value=features)
        return aggregated_feature.squeeze(1)

class MLPHead(nn.Module):
    """
    A simple MLP to map the aggregated feature vector to a final scalar score.
    """
    def __init__(self, input_dim: int, hidden_dim_mult: int = 4):
        super().__init__()
        hidden_dim = input_dim * hidden_dim_mult
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, cond_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + cond_dim * 2, out_dim)
        # self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], spks: float["b n d"], drop_audio_cond=False):  # noqa: F722
    # def forward(self, x: float["b n d"], cond: float["b n d"], timbre_cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
            spks = torch.zeros_like(spks)

        x = self.proj(torch.cat((x, cond, spks), dim=-1))
        # x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT_dis(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        bn_dim=256,
        qk_norm=None,
        conv_layers=0,
        chunk_size=8,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.t_time_embed = TimestepEmbedding(dim)
        self.r_time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, bn_dim, dim)
        self.cache_embed = nn.Linear(mel_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        
        self.timbre_encoder = MRTE(n_head=4, n_feat=bn_dim, dropout_rate=0., q_in_dim=bn_dim, k_in_dim=mel_dim, v_in_dim=mel_dim, num_blocks=2)
        
        self.transformer_blocks = nn.ModuleList(
            [
                ChunkDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    chunk_size=chunk_size,
                    pe_attn_head=pe_attn_head,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        # self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        # self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.discriminator_target_layers = [1, 3]

        self.cross_attentions = nn.ModuleList(
                [AttentionAggregation(dim=dim, num_heads=heads) for _ in self.discriminator_target_layers]
            )

        mlp_input_dim = dim * len(self.discriminator_target_layers)

        self.mlp_head = MLPHead(input_dim=mlp_input_dim)



        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.norm_out.linear.weight, 0)
        # nn.init.constant_(self.norm_out.linear.bias, 0)
        # nn.init.constant_(self.proj_out.weight, 0)
        # nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward


    def forward(
        self,
        x: float["b n d"],  # nosied input mel  # noqa: F722           B, T, 80
        t: float["b"] | float[""],  # time step  # noqa: F821 F722
        r: float["b"] | float[""],  # time step  # noqa: F821 F722
        cache: float["b n d"],
        cond: float["b n d"],  # bn  # noqa: F722         B, T, 256
        spks: float["b n d"],  # spks  # noqa: F722       B, T, 256
        prompts: float["b n d"],  # mel prompts  # noqa: F722       B, NT, 256
        offset=0,
        mask: bool["b n"] | None = None,  # noqa: F722
        is_inference: bool = False,
        is_uncondition: bool = False,
        cfg_mask: bool["b"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        spks_ = spks.unsqueeze(1).repeat(1, cond.shape[1], 1)
        
        t = self.t_time_embed(t)
        r = self.r_time_embed(r)
        t = t + r
        
        # add timbre encoding
        timbre_cond = self.timbre_encoder(cond, prompts, spks)   # B,T, bn_dim
        
        if cfg_mask is not None:
            cfg_mask_ = rearrange(cfg_mask, "b -> b 1 1") 
            timbre_cond = torch.where(cfg_mask_, torch.zeros_like(timbre_cond), timbre_cond)
            spks_ = torch.where(cfg_mask_, torch.zeros_like(spks_), spks_)
        
        x = self.input_embed(x, timbre_cond, spks_, drop_audio_cond=is_uncondition)
        
        
        if not is_inference:
            if cache != None:
                cache = self.cache_embed(cache)
                x = torch.concat((cache, x), dim=1)   # [b, 2n, dim]
                if mask is not None:
                    mask = torch.concat((mask, mask), dim=1)  # [b, 2n]
                
                cache_len = cache.shape[1]
                rope_cache = self.rotary_embed.forward_from_seq_len(cache_len)
                rope_x = self.rotary_embed.forward_from_seq_len(seq_len)
                rope = (torch.concat((rope_cache[0], rope_x[0]), dim=1), rope_cache[1])  
            else:
                rope = self.rotary_embed.forward_from_seq_len(seq_len)
            
        else:
            if cache != None:
                cache = self.cache_embed(cache)
                x = torch.concat((cache, x), dim=1)   # [b, cache_len + seq_len, dim]
                
                # inference does not need to consider mask
                cache_len = cache.shape[1]
                rope_cache = self.rotary_embed.forward_from_seq_len(cache_len)
                rope_x = self.rotary_embed.forward_from_seq_len(offset + seq_len)
                rope = (torch.concat((rope_cache[0], rope_x[0][:, -seq_len:, :]), dim=1), rope_cache[1])
            else:
                rope = self.rotary_embed.forward_from_seq_len(seq_len)
                

        if self.long_skip_connection is not None:
            residual = x


        aggregated_features = []
        for index_block, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope, is_inference=is_inference)

                if index_block in self.discriminator_target_layers:
                    attn_layer_idx = self.discriminator_target_layers.index(index_block)
                    attn_layer = self.cross_attentions[attn_layer_idx]

                    kv_features = x[:, -seq_len:, :] 
                    agg_feat = attn_layer(kv_features)
                    aggregated_features.append(agg_feat)

        final_agg_vector = torch.cat(aggregated_features, dim=-1) # (b, dim * num_target_layers)
        score = self.mlp_head(final_agg_vector) # (b, 1)


        # if self.long_skip_connection is not None:
        #     x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        
        
        # x = x[:, -seq_len:, :]
        # x = self.norm_out(x, t)
        
        # output = self.proj_out(x)
        
        return score



