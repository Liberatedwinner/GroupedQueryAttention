from typing import Tuple, Union, Optional
from einops import rearrange, einsum

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, constant_
from torch.nn.functional import softmax


class SoftmaxOne(nn.Module):
    """
    https://www.evanmiller.org/attention-is-off-by-one.html
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
            x: Tensor,
            dim: int = -1,
            keepdim: bool = True
    ) -> Tensor:
        # for numerical stability, take max; not mean
        max_values = torch.max(x, dim, keepdim=keepdim)[0]
        exp_applied = torch.exp(x - max_values)
        denominator = 1 + torch.sum(exp_applied, dim, keepdim=keepdim)

        return exp_applied / denominator


def softmax_one(
        x: Tensor,
        dim: int = -1,
        keep_dim: bool = True
) -> Tensor:
    return SoftmaxOne()(x, dim, keep_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self,
            query_heads: int = 8,
            kv_heads: int = 4,
            dim_head: Optional[int] = 64,
            dropout_rate: float = 0.1,
            bias: bool = True,
            use_layer_norm: bool = True,
            layer_norm_eps: float = 1e-5,
            gamma_init: float = 1.0,
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        dim_head = query_heads ** 2 if dim_head is None else dim_head

        # assertion checks
        assert query_heads % kv_heads == 0, (
            f'`query_heads`: {query_heads}; `kv_heads`: {kv_heads},'
            'Cannot make head groups!'
            'need to be query_heads = (some number) * kv_heads'
        )
        assert dim_head % 8 == 0, f'Please check whether dim_head ({dim_head}) = (some number) * 8'
        assert dim_head <= 128, f'Please check whether dim_head ({dim_head}) <= 128'
        # prevent memory exploding

        embed_dim = query_heads * dim_head
        kv_embed_dim = kv_heads * dim_head
        setting_dict = {'device': device, 'dtype': dtype}

        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.num_head_groups = query_heads // kv_heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_layer_norm = use_layer_norm
        self.gamma_init = gamma_init

        self.attn_norm = nn.LayerNorm(
            embed_dim, eps=layer_norm_eps, **setting_dict
        )
        self.layer_norm = nn.LayerNorm(
            kv_embed_dim, eps=layer_norm_eps, **setting_dict
        ) if use_layer_norm else None

        setting_dict['bias'] = bias
        self.fc_q = nn.Linear(embed_dim, embed_dim, **setting_dict)
        self.fc_k = nn.Linear(embed_dim, kv_embed_dim, **setting_dict)
        self.fc_v = nn.Linear(embed_dim, kv_embed_dim, **setting_dict)
        self.fc_out = nn.Linear(kv_embed_dim, embed_dim, **setting_dict)

        self.group_pattern = ''

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.fc_q, self.fc_k]:
            xavier_normal_(module.weight)
            if module.bias is not None:
                constant_(module.bias, val=0)

        for module_w_gain in [self.fc_v, self.fc_out]:
            xavier_normal_(module_w_gain.weight, gain=self.gamma_init)
            if module_w_gain.bias is not None:
                constant_(module_w_gain.bias, val=0)

    def _compute_scaled_similarity(
            self,
            q: Tensor,
            k: Tensor,
            scale: Optional[float] = None,
            force_grouped: bool = False,
    ) -> Tensor:
        scale = q.shape[-1] ** (- 0.5) if scale is None else scale
        if self.num_head_groups > 1 or force_grouped:
            q = rearrange(
                q, 'b (h_kv g) n d -> b g h_kv n d', g=self.num_head_groups
            )
            self.group_pattern = 'g h_kv'
        else:
            self.group_pattern = 'h_kv'

        similarity = einsum(q, k, f'b {self.group_pattern} n d, b h_kv s d -> b {self.group_pattern} n s')  # WLOG, 5-dim tensor if `num_head_groups` > 1
        similarity *= scale

        return similarity

    def _masking_similarity(
            self,
            q: Tensor,
            k: Tensor,
            similarity: Tensor,
            mask: Optional[Tensor] = None,
            use_lower_tri_attn: bool = False,
            use_softmax_one: bool = False
    ) -> Tensor:
        batch_size_q, _, seq_length_q, _ = q.shape
        seq_length_kv = k.shape[2]
        mask = torch.ones(
            (batch_size_q, seq_length_q, seq_length_kv),
            device=q.device,
            dtype=torch.bool
        ).tril_() if use_lower_tri_attn else mask

        if mask is not None:
            if mask.ndim == 2:
                # print('Expand the shape of mask, two times')
                mask = rearrange(mask, 'b s -> b () () () s')
            elif mask.ndim == 3:
                # print('Expand the shape of mask, one time')
                mask = rearrange(mask, 'b n s -> b () () n s')
            similarity.masked_fill_(~mask.bool(), torch.finfo(similarity.dtype).min)

        softmax_ftn = softmax_one if use_softmax_one else softmax
        similarity = softmax_ftn(similarity, dim=-1)
        similarity = self.dropout(similarity)

        return similarity

    def forward(
            self,
            input_tensor: Tensor,
            input_for_multi: Optional[Tensor] = None,
            scale: Optional[float] = None,
            mask: Optional[Tensor] = None,
            use_lower_tri_attn: bool = False,
            use_softmax_one: bool = True,
            force_grouped: bool = False,
            need_weights: bool = False,
            average_attn_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        in patterns of `rearrange`,
            b: batch size
            n: length of sequence q
            s: length of sequence k, v
            h: number of heads; #(q_heads) != #(k_heads) = #(v_heads)
            d: the dimension of heads
        """
        input_tensor = self.attn_norm(input_tensor)
        input_for_multi = input_tensor if input_for_multi is None else input_for_multi
        dim_split_pattern = 'b n (h d) -> b h n d'  # split and transpose

        q = self.fc_q(input_tensor)
        k, v = map(
            lambda ftn: ftn(input_for_multi), (self.fc_k, self.fc_v)
        )
        q = rearrange(
            q, dim_split_pattern, h=self.query_heads
        )
        k, v = map(
            lambda z: rearrange(z, dim_split_pattern, h=self.kv_heads),
            (k, v)
        )
        similarity = self._compute_scaled_similarity(q, k, scale, force_grouped)
        print(f'group pattern is: {self.group_pattern}')
        similarity = self._masking_similarity(
            q,
            k,
            similarity,
            mask,
            use_lower_tri_attn,
            use_softmax_one
        )

        attention = einsum(similarity, v, f'b {self.group_pattern} n s, b h_kv s d -> b {self.group_pattern} n d')
        orig_pattern = ' '.join(self.group_pattern.split()[::-1])  # flatten order: head → group 
        x = rearrange(attention, f'b {self.group_pattern} n d -> b n ({orig_pattern}) d')  

        attn_weights = None
        if need_weights:
            attn_weights = rearrange(attention, f'b {self.group_pattern} n s -> b n s ({orig_pattern})')
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)

        x = rearrange(x, 'b n h d -> b n (h d)')
        x = self.layer_norm(x) if self.use_layer_norm else x
        out = self.fc_out(x)

        return out, attn_weights
