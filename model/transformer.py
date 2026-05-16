import torch
import torch.nn as nn
from einops import rearrange

try:
    import xformers.ops as xops
except ImportError as exc:
    raise ImportError("Please install xformers to use flash attention") from exc


def init_weights(module, std=0.02):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight.type_as(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, bias=False, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class IndependentVAttention(nn.Module):
    """Shared Q/K routing with independent semantic and spatial value streams."""

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for semantic/spatial branches"
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.d_half = dim // 2
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout

        self.to_qk = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.to_v_semantic = nn.Linear(self.d_half, self.d_half, bias=qkv_bias)
        self.to_v_spatial = nn.Linear(self.d_half, self.d_half, bias=qkv_bias)
        self.fc_semantic = nn.Linear(self.d_half, self.d_half, bias=fc_bias)
        self.fc_spatial = nn.Linear(self.d_half, self.d_half, bias=fc_bias)
        self.dropout = nn.Dropout(fc_dropout)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None):
        semantic, spatial = x.split(self.d_half, dim=-1)
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = torch.cat([self.to_v_semantic(semantic), self.to_v_spatial(spatial)], dim=-1)

        q, k, v = (
            rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = xops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            p=self.attn_dropout if self.training else 0.0,
            op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
        )
        out = rearrange(out, "b l nh dh -> b l (nh dh)")
        semantic, spatial = out.split(self.d_half, dim=-1)
        return self.dropout(torch.cat([self.fc_semantic(semantic), self.fc_spatial(spatial)], dim=-1))


class DecoupledTransformerBlock(nn.Module):
    """Semantic-spatial decoupled Transformer block with optional bidirectional modulation."""

    def __init__(
        self,
        dim,
        head_dim,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_mod=False,
        mod_scale_init=1.0,
    ):
        super().__init__()
        self.d_half = dim // 2
        self.use_mod = use_mod

        self.norm_semantic_1 = nn.LayerNorm(self.d_half, bias=ln_bias)
        self.norm_spatial_1 = nn.LayerNorm(self.d_half, bias=ln_bias)
        self.norm_semantic_2 = nn.LayerNorm(self.d_half, bias=ln_bias)
        self.norm_spatial_2 = nn.LayerNorm(self.d_half, bias=ln_bias)
        self.attn = IndependentVAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
        )
        self.mlp_semantic = MLP(self.d_half, mlp_ratio=mlp_ratio, bias=mlp_bias, dropout=mlp_dropout)
        self.mlp_spatial = MLP(self.d_half, mlp_ratio=mlp_ratio, bias=mlp_bias, dropout=mlp_dropout)

        if use_mod:
            self.mod_spatial_to_semantic = nn.Linear(self.d_half, 2 * self.d_half)
            self.mod_semantic_to_spatial = nn.Linear(self.d_half, 2 * self.d_half)
            self.mod_norm_spatial = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.mod_norm_semantic = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.reset_mod_parameters(mod_scale_init)

    def reset_mod_parameters(self, mod_scale_init=1.0):
        if not self.use_mod:
            return
        for generator in (self.mod_spatial_to_semantic, self.mod_semantic_to_spatial):
            nn.init.zeros_(generator.weight)
            with torch.no_grad():
                generator.bias[: self.d_half].fill_(mod_scale_init)
                generator.bias[self.d_half :].zero_()

    def _apply_modulation(self, semantic, spatial):
        gamma_beta = self.mod_spatial_to_semantic(self.mod_norm_spatial(spatial))
        gamma, beta = gamma_beta.split(self.d_half, dim=-1)
        semantic = gamma * semantic + beta

        gamma_beta = self.mod_semantic_to_spatial(self.mod_norm_semantic(semantic))
        gamma, beta = gamma_beta.split(self.d_half, dim=-1)
        spatial = gamma * spatial + beta
        return semantic, spatial

    def forward(self, x):
        semantic, spatial = x.split(self.d_half, dim=-1)
        h = torch.cat([self.norm_semantic_1(semantic), self.norm_spatial_1(spatial)], dim=-1)
        x = x + self.attn(h)

        semantic, spatial = x.split(self.d_half, dim=-1)
        if self.use_mod:
            semantic, spatial = self._apply_modulation(semantic, spatial)
            x = torch.cat([semantic, spatial], dim=-1)

        semantic, spatial = x.split(self.d_half, dim=-1)
        semantic = self.mlp_semantic(self.norm_semantic_2(semantic))
        spatial = self.mlp_spatial(self.norm_spatial_2(spatial))
        return x + torch.cat([semantic, spatial], dim=-1)
