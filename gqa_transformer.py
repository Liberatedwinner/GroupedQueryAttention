from typing import Union, Optional, Callable

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, constant_

from grouped_query_attentions import GroupedQueryAttention


def compute_scale_constants(
        num_encoders: Optional[int], num_decoders: Optional[int]
) -> Tuple[int, float]:
    # from MAGNETO paper
    assert not (num_encoders is None and num_decoders is None)

    if num_decoders is None:
        encoder_scale_const = log(2 * num_encoders) ** 0.5
        decoder_scale_const = 0

    elif num_encoders is None:
        encoder_scale_const = 0
        decoder_scale_const = log(2 * num_decoders) ** 0.5

    else:
        sub_e = log(2 * num_encoders) / 3
        sub_d = log(3 * num_decoders)
        encoder_scale_const = (sub_d * sub_e) ** 0.5
        decoder_scale_const = sub_d ** 0.5

    return encoder_scale_const, decoder_scale_const


class GQAEncoder(nn.Module):
    def __init__(
            self,
            query_heads: int = 8, kv_heads: int = 4,
            dim_head: Optional[int] = None,
            dim_hidden: int = 3072, dropout_rate: float = 0.1,
            bias: bool = True,
            layer_norm_eps: float = 1e-5, gamma_init: float = 0.02,
            act_ftn: Union[str, Callable] = 'gelu',
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        act_dict = {'gelu': nn.GELU()}
        attn_param_dict = dict(
            query_heads=query_heads,
            kv_heads=kv_heads,
            dim_head=dim_head,
            dropout_rate=dropout_rate,
            bias=bias,
            use_layer_norm=True,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype
        )
        self.gamma_init = gamma_init
        self.attn = GroupedQueryAttention(**attn_param_dict)
        self.act_ftn = act_dict[act_ftn] if isinstance(act_ftn, str) else act_ftn
        self.dropout = nn.Dropout(dropout_rate)

        embed_dim = query_heads * dim_head
        setting_dict = dict(device=device, dtype=dtype)
        self.upper_layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **setting_dict)
        self.lower_layer_norm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps, **setting_dict)

        setting_dict['bias'] = bias
        self.upper_fc = nn.Linear(embed_dim, dim_hidden, **setting_dict)
        self.lower_fc = nn.Linear(dim_hidden, embed_dim, **setting_dict)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.upper_fc, self.lower_fc]:
            xavier_normal_(module.weight, gain=self.gamma_init)
            if module.bias is not None:
                constant_(module.bias, val=0)

    def _apply_attention(
            self, x: Tensor, mask: Optional[Tensor] = None, use_lta: bool = False
    ) -> Tensor:
        x, _ = self.attn(x, mask=mask, use_lower_tri_attn=use_lta)  # first Layer normalization + Attention
        out = self.dropout(x)

        return out

    def _make_sequential_layer(self, x: Tensor) -> Tensor:
        sequential_layer = nn.Sequential(
            self.upper_layer_norm, self.upper_fc,
            self.act_ftn,
            self.dropout,
            self.lower_layer_norm, self.lower_fc,
            self.dropout
        )

        return sequential_layer(x)

    def forward(
            self, input_tensor: Tensor, input_mask: Optional[Tensor] = None, use_lta: bool = False
    ) -> Tensor:
        x = input_tensor + self._apply_attention(input_tensor, mask=input_mask, use_lta=use_lta)
        out = x + self._make_sequential_layer(x)
        # `+=` is not used; this verbose usage prevents CUDA device-side assertion error

        return out


class GQADecoder(nn.Module):
    def __init__(
            self,
            query_heads: int = 8, kv_heads: int = 4,
            dim_head: Optional[int] = None,
            dim_hidden: int = 3072, dropout_rate: float = 0.1,
            bias: bool = True,
            layer_norm_eps: float = 1e-5, gamma_init: float = 0.02,
            act_ftn: Union[str, Callable] = 'gelu',
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        act_dict = {'gelu': nn.GELU()}
        # TODO: get as dict itself?
        attn_param_dict = dict(
            query_heads=query_heads,
            kv_heads=kv_heads,
            dim_head=dim_head,
            dropout_rate=dropout_rate,
            bias=bias,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype
        )
        self.gamma_init = gamma_init
        self.act_ftn = act_dict[act_ftn] if isinstance(act_ftn, str) else act_ftn
        self.dropout = nn.Dropout(dropout_rate)
        self.self_attn = GroupedQueryAttention(**attn_param_dict)
        self.multi_attn = GroupedQueryAttention(use_layer_norm=True, **attn_param_dict)

        embed_dim = query_heads * dim_head
        setting_dict = dict(device=device, dtype=dtype)
        self.upper_layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **setting_dict)
        self.lower_layer_norm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps, **setting_dict)

        setting_dict['bias'] = bias
        self.upper_fc = nn.Linear(embed_dim, dim_hidden, **setting_dict)
        self.lower_fc = nn.Linear(dim_hidden, embed_dim, **setting_dict)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.upper_fc, self.lower_fc]:
            xavier_normal_(module.weight, gain=self.gamma_init)
            if module.bias is not None:
                constant_(module.bias, val=0)

    def _apply_self_attention(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            use_lta: bool = False
    ) -> Tensor:
        x, _ = self.self_attn(x, mask=mask, use_lower_tri_attn=use_lta)
        out = self.dropout(x)

        return out

    def _apply_multi_attention(
            self,
            x1: Tensor,
            x2: Tensor,
            x2_mask: Optional[Tensor] = None,
            use_lta: bool = False
    ) -> Tensor:
        x1, _ = self.multi_attn(x1, x2, mask=x2_mask, use_lower_tri_attn=use_lta)
        out = self.dropout(x1)

        return out

    def _make_sequential_layer(self, x: Tensor) -> Tensor:
        sequential_layer = nn.Sequential(
            self.upper_layer_norm, self.upper_fc,
            self.act_ftn,
            self.dropout,
            self.lower_layer_norm, self.lower_fc,
            self.dropout
        )

        return sequential_layer(x)

    def forward(
            self,
            input_tensor: Tensor,
            memory_tensor: Tensor,
            input_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            use_lta_self: bool = False,
            use_lta_memory: bool = False
    ):
        x = input_tensor + self._apply_self_attention(input_tensor, mask=input_mask, use_lta=use_lta_self)
        x = x + self._apply_multi_attention(x, memory_tensor, x2_mask=memory_mask, use_lta=use_lta_memory)
        out = x + self._make_sequential_layer(x)

        return out


class GQATransformer(nn.Module):
    def __init__(
            self,
            query_heads: int = 8, kv_heads: int = 4, dim_head: Optional[int] = 32,
            num_encoder_layers: int = 12, num_decoder_layers: Optional[int] = 12,
            dim_hidden: int = 3072, dropout_rate: float = 0.1,
            layer_norm_eps: float = 1e-12,
            act_ftn: Union[str, Callable] = 'gelu',
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        encoder_scale_const, decoder_scale_const = compute_scale_constants(
            num_encoder_layers, num_decoder_layers
        )
        common_params_dict = dict(
            query_heads=query_heads,
            kv_heads=kv_heads,
            dim_head=dim_head,
            dim_hidden=dim_hidden,
            dropout_rate=dropout_rate,
            layer_norm_eps=layer_norm_eps,
            act_ftn=act_ftn,
            device=device,
            dtype=dtype
        )

        self.encoder_layers = nn.ModuleList([
            GQAEncoder(
                gamma_init=encoder_scale_const,
                **common_params_dict
            ) for _ in range(num_encoder_layers)
        ])

        if num_decoder_layers is not None:
            self.decoder_layers = nn.ModuleList([
                GQADecoder(
                    gamma_init=decoder_scale_const,
                    **common_params_dict
                ) for _ in range(num_decoder_layers)
            ])
        else:
            self.decoder_layers = None

    def forward(
            self,
            x: Tensor,
            input_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            use_lta_self: bool = False,
            use_lta_memory: bool = False
    ) -> Tensor:
        x_d = x  # TODO: deepcopy?
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, input_mask=input_mask, use_lta=use_lta_self)

        if self.decoder_layers is None:
            out = x

        else:
            encoder_memory = x

            # TODO: need to check the action of this part...
            for decoder_layer in self.decoder_layers:
                x_d = decoder_layer(
                    x_d,
                    encoder_memory,
                    input_mask=input_mask,
                    memory_mask=memory_mask,
                    use_lta_self=use_lta_self,
                    use_lta_memory=use_lta_memory
                )
            out = x_d

        return out
