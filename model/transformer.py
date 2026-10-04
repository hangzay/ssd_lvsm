import torch
import torch.nn as nn
from einops import rearrange

from .camera_conditioning import ProjectivePositionEncoding

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
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


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
        projective_position_encoding=None,
    ):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for semantic/spatial branches"
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.d_half = dim // 2
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout
        self.projective_position_encoding = projective_position_encoding

        self.to_qk = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.to_vI = nn.Linear(self.d_half, self.d_half, bias=qkv_bias)
        self.to_vP = nn.Linear(self.d_half, self.d_half, bias=qkv_bias)
        self.fc_I = nn.Linear(self.d_half, self.d_half, bias=fc_bias)
        self.fc_P = nn.Linear(self.d_half, self.d_half, bias=fc_bias)
        self.dropout = nn.Dropout(fc_dropout)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None, camera_context=None):
        semantic, spatial = x.split(self.d_half, dim=-1)
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = torch.cat([self.to_vI(semantic), self.to_vP(spatial)], dim=-1)

        q, k, v = (
            rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        if camera_context is None:
            out = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            if attn_bias is not None:
                raise ValueError("PRoPE/GTA does not currently support an attention bias.")
            if self.projective_position_encoding is None:
                raise ValueError("Camera context was provided but projective attention is not initialized.")
            out = self.projective_position_encoding(
                rearrange(q, "b l nh dh -> b nh l dh"),
                rearrange(k, "b l nh dh -> b nh l dh"),
                rearrange(v, "b l nh dh -> b nh l dh"),
                camera_context,
                dropout_p=self.attn_dropout if self.training else 0.0,
                token_features=x,
            )
            out = rearrange(out, "b nh l dh -> b l nh dh")
        out = rearrange(out, "b l nh dh -> b l (nh dh)")
        semantic, spatial = out.split(self.d_half, dim=-1)
        return self.dropout(torch.cat([self.fc_I(semantic), self.fc_P(spatial)], dim=-1))


class StandardSelfAttention(nn.Module):
    """Original LVSM attention used when semantic-spatial decoupling is disabled."""

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        projective_position_encoding=None,
    ):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout
        self.projective_position_encoding = projective_position_encoding
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.dropout = nn.Dropout(fc_dropout)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None, camera_context=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
            for t in (q, k, v)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        if camera_context is None:
            out = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_dropout if self.training else 0.0,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            if attn_bias is not None:
                raise ValueError("PRoPE/GTA does not currently support an attention bias.")
            if self.projective_position_encoding is None:
                raise ValueError("Camera context was provided but projective attention is not initialized.")
            out = self.projective_position_encoding(
                rearrange(q, "b l nh dh -> b nh l dh"),
                rearrange(k, "b l nh dh -> b nh l dh"),
                rearrange(v, "b l nh dh -> b nh l dh"),
                camera_context,
                dropout_p=self.attn_dropout if self.training else 0.0,
                token_features=x,
            )
            out = rearrange(out, "b nh l dh -> b l nh dh")
        return self.dropout(self.fc(rearrange(out, "b l nh dh -> b l (nh dh)")))


class DecoupledTransformerBlock(nn.Module):
    """LVSM block with one baseline/decouple switch and the final bidirectional FiLM."""

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
        decouple=True,
        film=False,
        film_scale_init=1.0,
        attention_encoding="none",
        patches_x=None,
        patches_y=None,
        prope_freq_base=100.0,
        prope_freq_scale=1.0,
    ):
        super().__init__()
        if film and not decouple:
            raise ValueError("film=True requires decouple=True")
        self.d_half = dim // 2
        self.decouple = decouple
        self.film = film
        if attention_encoding not in ("none", "prope", "gta"):
            raise ValueError(f"Unsupported attention encoding '{attention_encoding}'.")
        projective_position_encoding = None
        if attention_encoding != "none":
            if patches_x is None or patches_y is None:
                raise ValueError("Camera attention encoding requires a fixed patch grid.")
            projective_position_encoding = ProjectivePositionEncoding(
                head_dim=head_dim,
                patches_x=patches_x,
                patches_y=patches_y,
                freq_base=prope_freq_base,
                freq_scale=prope_freq_scale,
            )

        if decouple:
            self.normI1 = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.normP1 = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.normI2 = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.normP2 = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.attn = IndependentVAttention(
                dim=dim,
                head_dim=head_dim,
                qkv_bias=attn_qkv_bias,
                fc_bias=attn_fc_bias,
                attn_dropout=attn_dropout,
                fc_dropout=attn_fc_dropout,
                projective_position_encoding=projective_position_encoding,
            )
            self.mlp_I = MLP(self.d_half, mlp_ratio=mlp_ratio, bias=mlp_bias, dropout=mlp_dropout)
            self.mlp_P = MLP(self.d_half, mlp_ratio=mlp_ratio, bias=mlp_bias, dropout=mlp_dropout)
        else:
            self.norm_1 = nn.LayerNorm(dim, bias=ln_bias)
            self.norm_2 = nn.LayerNorm(dim, bias=ln_bias)
            self.attn = StandardSelfAttention(
                dim=dim,
                head_dim=head_dim,
                qkv_bias=attn_qkv_bias,
                fc_bias=attn_fc_bias,
                attn_dropout=attn_dropout,
                fc_dropout=attn_fc_dropout,
                projective_position_encoding=projective_position_encoding,
            )
            self.mlp = MLP(dim, mlp_ratio=mlp_ratio, bias=mlp_bias, dropout=mlp_dropout)

        if film:
            self.film_generator = nn.Linear(self.d_half, 2 * self.d_half)
            self.film_generator_i2p = nn.Linear(self.d_half, 2 * self.d_half)
            self.film_norm_P = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.film_norm_I = nn.LayerNorm(self.d_half, bias=ln_bias)
            self.reset_film_parameters(film_scale_init)

    def reset_film_parameters(self, film_scale_init=1.0):
        if not self.film:
            return
        for generator in (self.film_generator, self.film_generator_i2p):
            nn.init.zeros_(generator.weight)
            with torch.no_grad():
                generator.bias[: self.d_half].fill_(film_scale_init)
                generator.bias[self.d_half :].zero_()


    def _apply_film(self, semantic, spatial):
        gamma_beta = self.film_generator(self.film_norm_P(spatial))
        gamma, beta = gamma_beta.split(self.d_half, dim=-1)
        semantic = gamma * semantic + beta

        gamma_beta = self.film_generator_i2p(self.film_norm_I(semantic))
        gamma, beta = gamma_beta.split(self.d_half, dim=-1)
        spatial = gamma * spatial + beta
        return semantic, spatial

    def forward(self, x, camera_context=None):
        if not self.decouple:
            x = x + self.attn(self.norm_1(x), camera_context=camera_context)
            return x + self.mlp(self.norm_2(x))

        semantic, spatial = x.split(self.d_half, dim=-1)
        h = torch.cat([self.normI1(semantic), self.normP1(spatial)], dim=-1)
        x = x + self.attn(h, camera_context=camera_context)

        semantic, spatial = x.split(self.d_half, dim=-1)
        if self.film:
            semantic, spatial = self._apply_film(semantic, spatial)
            x = torch.cat([semantic, spatial], dim=-1)

        semantic, spatial = x.split(self.d_half, dim=-1)
        semantic = self.mlp_I(self.normI2(semantic))
        spatial = self.mlp_P(self.normP2(spatial))
        return x + torch.cat([semantic, spatial], dim=-1)
