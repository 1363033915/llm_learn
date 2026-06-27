"""
Lesson 2 Extended: 从零手写 Qwen3.5-0.8B — 端到端推理

Qwen3.5-0.8B 是混合架构模型（Gated DeltaNet + GQA Full Attention）：
  - 24 层 Decoder，其中 6 层 full_attention、18 层 linear_attention
  - full_attention（每 4 层一个）：GQA + QK-Norm + Partial RoPE + Output Gate
  - linear_attention：Gated DeltaNet（纯 PyTorch 递推，无需 FLA / causal-conv1d）

用法:
    python run_qwen3_5_0_8B.py --model D:\\code\\model\\Qwen3.5-0.8B
"""

import time
import argparse
import json
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# RMSNorm — Qwen3.5 风格（weight 初始化为 0，公式为 x * (1 + weight)）
# ============================================================================
#
# 与标准 RMSNorm 的区别：
#   标准:       y = x / RMS(x) * weight           (weight 初始化为 1)
#   Qwen3.5:   y = x / RMS(x) * (1 + weight)      (weight 初始化为 0)
# 这样初始化时 weight=0 → (1+0)=1，输出就是归一化后的值，不额外缩放。

class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # 初始化为 0！

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())  # (1 + weight)，不是 weight
        return output.type_as(x)


# ============================================================================
# RMSNormGated — 带门控的 RMSNorm
# ============================================================================
#
# 用于 GatedDeltaNet 的 value 输出归一化：
#   y = norm(x) * silu(gate)
#
# weight 初始化为 1（普通 RMSNorm 风格，不是 1+weight）

class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.float())  # 门控
        return x.to(input_dtype)


# ============================================================================
# Partial Rotary Position Embedding
# ============================================================================
#
# Qwen3.5 的 full_attention 使用 partial RoPE：
#   - head_dim = 256, partial_rotary_factor = 0.25
#   - 只对前 256 * 0.25 = 64 维做 RoPE，后 192 维不旋转
#   - base = 10,000,000（Qwen3 是 1,000,000，这里更大）
#
# 因为是纯文本推理，不需要 3D MRoPE（T/H/W 位置相同，退化为 1D）

class Qwen3_5RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, partial_rotary_factor: float = 0.25,
                 base: float = 10000000.0):
        super().__init__()
        # 只对部分维度做 RoPE
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        # theta_i = 1 / base^(2i/d), i = 0, 1, ..., rotary_dim/2-1
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (batch, seq_len) — 纯文本位置 [0, 1, 2, ..., S-1]
        returns: cos, sin, shape (1, seq_len, rotary_dim)
        """
        # freqs = position * inv_freq, (seq_len, rotary_dim/2)
        freqs = torch.outer(position_ids[0].float(), self.inv_freq)
        # 复制拼成完整 rotary_dim: (seq_len, rotary_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """交换前后半并取负: [x1, x2] -> [-x2, x1]"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q/K 应用 Partial RoPE（只旋转前 rotary_dim 维）

    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (1, seq_len, rotary_dim)
    """
    cos = cos.unsqueeze(1).to(q.dtype)   # (1, 1, seq_len, rotary_dim)
    sin = sin.unsqueeze(1).to(q.dtype)

    # 拆分：前 rotary_dim 维做 RoPE，后面不做
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


# ============================================================================
# Grouped Query Attention (GQA) + QK-Norm + Partial RoPE + Output Gate
# ============================================================================
#
# Qwen3.5 的 full_attention 有 4 个特有设计：
#   1. QK-Norm（在 RoPE 之前对每个头的 Q/K 做 RMSNorm）
#   2. Partial RoPE（只旋转 head_dim 的前 partial_rotary_factor 部分）
#   3. Output Gate（q_proj 输出 2x 维度，用 sigmoid 门控 attention 输出）
#   4. Qwen3_5RMSNorm（weight 初始化为 0，公式 1+weight）

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: 复制 KV 头以匹配 Q 头数"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        # q_proj 输出 2x：前一半为 query，后一半为 output gate
        self.q_proj = nn.Linear(config.hidden_size,
                                self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size,
                                self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size,
                                self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                config.hidden_size, bias=False)

        # QK-Norm（在 RoPE 之前，对每个头的 head_dim 做 RMSNorm）
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,            # (batch, seq_len, hidden_size)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        attention_mask: torch.Tensor,            # (1, 1, seq_len, seq_len)
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Q 投影并拆分为 query_states 和 gate
        q_and_gate = self.q_proj(hidden_states)
        q_and_gate = q_and_gate.view(B, S, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        # gate: (B, S, num_heads, head_dim) -> flatten for later
        gate = gate.reshape(B, S, self.num_heads * self.head_dim)

        # K, V 投影
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)

        # QK-Norm（在 reshape 成多头之后、RoPE 之前）
        query_states = self.q_norm(query_states).transpose(1, 2)  # (B, num_heads, S, head_dim)
        k = self.k_norm(k).transpose(1, 2)                         # (B, num_kv_heads, S, head_dim)
        v = v.transpose(1, 2)                                       # (B, num_kv_heads, S, head_dim)

        # Partial RoPE（只旋转前 rotary_dim 维）
        cos, sin = position_embeddings
        query_states, k = apply_partial_rotary_pos_emb(
            query_states, k, cos, sin, self.rotary_dim)

        # GQA：复制 KV 头
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 标准注意力
        attn_weights = torch.matmul(query_states, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # reshape 回 (B, S, hidden)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)

        # Output Gate: attn_output * sigmoid(gate)
        attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)


# ============================================================================
# Gated DeltaNet — Qwen3.5 的线性注意力
# ============================================================================
#
# Gated DeltaNet 是一种基于递推的线性注意力机制，核心思路：
#
#   1. 对输入做 causal conv1d 得到 q, k, v
#   2. 使用递推状态 S_t（key_dim × value_dim 矩阵）：
#      S_t = g_t * S_{t-1} + k_t ⊗ (β_t * (v_t - k_t^T S_{t-1}))
#   3. 输出 o_t = S_t * q_t
#
# 其中：
#   - g_t = exp(-exp(A_log) * softplus(a + dt_bias))  — 逐头衰减因子
#   - β_t = sigmoid(b)                                 — 逐头更新门控
#   - q_t, k_t 做了 L2 归一化 (l2norm)
#
# 这里用纯 PyTorch 实现递推版本，避免对 FLA / causal-conv1d 库的依赖。

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 归一化（对齐 FLA 库的实现）"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads    # 16
        self.num_v_heads = config.linear_num_value_heads  # 16
        self.head_k_dim = config.linear_key_head_dim      # 128
        self.head_v_dim = config.linear_value_head_dim    # 128
        self.key_dim = self.num_k_heads * self.head_k_dim   # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim # 2048
        self.conv_kernel = config.linear_conv_kernel_dim    # 4
        self.conv_dim = self.key_dim * 2 + self.value_dim   # 6144

        # QKV 投影（q, k, v 合并在一个投影中）
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)

        # Depthwise causal conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,       # depthwise
            padding=self.conv_kernel - 1,
        )

        # 输出门控投影
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # 衰减因子和更新门控
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # 可学习参数
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        # A_log: 初始化为 log(uniform(0, 16))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Value 输出归一化（RMSNormGated，由 in_proj_z 门控）
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        # 输出投影
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # ---- 1. 投影 QKV ----
        mixed_qkv = self.in_proj_qkv(hidden_states)     # (B, S, 6144)

        # ---- 2. Causal Conv1d ----
        # 转成 (B, conv_dim, S) 做 depthwise causal conv
        mixed_qkv = mixed_qkv.transpose(1, 2)            # (B, 6144, S)
        # self.conv1d 有 padding=3，截取前 S 个输出得到因果效果
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :S])
        mixed_qkv = mixed_qkv.transpose(1, 2)            # (B, S, 6144)

        # ---- 3. 拆分为 Q, K, V，reshape 成多头 ----
        q, k, v = torch.split(mixed_qkv,
                              [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, S, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, S, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, S, self.num_v_heads, self.head_v_dim)

        # ---- 4. 门控、衰减 ----
        z = self.in_proj_z(hidden_states)                # (B, S, value_dim)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)  # (B, S, 16, 128)

        a = self.in_proj_a(hidden_states)                # (B, S, 16)
        b = self.in_proj_b(hidden_states)                # (B, S, 16)

        beta = torch.sigmoid(b)                           # (B, S, 16) 更新门控
        # 衰减因子: g = -exp(A_log) * softplus(a + dt_bias) < 0
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g = g.to(hidden_states.dtype)                    # (B, S, 16)

        # ---- 5. Q/K L2 归一化 ----
        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)

        # ---- 6. Gated Delta Rule（纯 PyTorch 递推） ----
        # 转成 (B, num_heads, S, head_dim) 便于递推
        q = q.transpose(1, 2).contiguous()    # (B, 16, S, 128)
        k = k.transpose(1, 2).contiguous()    # (B, 16, S, 128)
        v = v.transpose(1, 2).contiguous()    # (B, 16, S, 128)
        # beta 和 g 需要扩展出头维度
        beta = beta.transpose(1, 2).contiguous()    # (B, 16, S)
        g = g.transpose(1, 2).contiguous()          # (B, 16, S)

        scale = 1.0 / (self.head_k_dim ** 0.5)
        q = q * scale

        # 递推状态 S: (B, num_v_heads, head_k_dim, head_v_dim)
        state = torch.zeros(B, self.num_v_heads,
                            self.head_k_dim, self.head_v_dim,
                            dtype=v.dtype, device=v.device)

        outputs = torch.zeros_like(v)

        for t in range(S):
            q_t = q[:, :, t, :]      # (B, 16, 128)
            k_t = k[:, :, t, :]      # (B, 16, 128)
            v_t = v[:, :, t, :]      # (B, 16, 128)
            g_t = g[:, :, t].exp()   # (B, 16)
            beta_t = beta[:, :, t]   # (B, 16)

            # 衰减
            state = state * g_t[:, :, None, None]  # (B, 16, 128, 128)

            # Delta rule: kv_mem = K^T @ S, delta = (v - kv_mem) * beta
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, 16, 128)
            delta = (v_t - kv_mem) * beta_t[:, :, None]        # (B, 16, 128)

            # 更新状态: S = S + k ⊗ delta
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, 16, 128, 128)

            # 输出: o = S @ q
            o_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)   # (B, 16, 128)
            outputs[:, :, t, :] = o_t

        outputs = outputs.transpose(1, 2).contiguous()  # (B, S, 16, 128)

        # ---- 7. RMSNormGated（norm(x) * silu(gate)） ----
        outputs = outputs.reshape(-1, self.head_v_dim)   # (B*S*16, 128)
        z = z.reshape(-1, self.head_v_dim)               # (B*S*16, 128)
        outputs = self.norm(outputs, z)                  # RMSNorm + silu(gate)
        outputs = outputs.reshape(B, S, self.value_dim)  # (B, S, 2048)

        # ---- 8. 输出投影 ----
        return self.out_proj(outputs)                    # (B, S, hidden_size)


# ============================================================================
# SwiGLU MLP
# ============================================================================
#
# 与标准 Qwen3 相同：down_proj(silu(gate_proj(x)) * up_proj(x))

class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Decoder Layer — 根据 layer_types 分派到 linear 或 full attention
# ============================================================================

class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-Norm: x → LN → Attn → + → LN → MLP → +
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(
                hidden_states, position_embeddings, attention_mask
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Qwen3.5 Text Model — Transformer 骨干
# ============================================================================

class Qwen3_5TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            head_dim=config.head_dim,
            partial_rotary_factor=config.partial_rotary_factor,
            base=config.rope_theta,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        # 位置 ID
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # RoPE cos/sin（只对有 rotary 的 head_dim 部分）
        cos, sin = self.rotary_emb(position_ids)
        position_embeddings = (cos, sin)

        # 因果掩码（仅 full_attention 层使用）
        causal_mask = torch.full((S, S), float("-inf"), device=input_ids.device,
                                 dtype=hidden_states.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)

        return self.norm(hidden_states)


# ============================================================================
# Qwen3.5 ForCausalLM — 完整语言模型
# ============================================================================

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


# ============================================================================
# 权重加载 — 从本地 safetensors 文件加载
# ============================================================================
#
# Qwen3.5-0.8B 是多模态模型，权重前缀为 model.language_model.*。
# 我们只加载文本部分（跳过 model.visual.* 和 mtp.*）。
# lm_head 没有独立权重，tie_word_embeddings=True → 与 embed_tokens 共享。

def load_weights_from_safetensors(model: Qwen3_5ForCausalLM, model_path: str,
                                   device: torch.device, dtype: torch.dtype):
    """从 safetensors 文件加载文本模型权重"""
    import safetensors

    files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    if not files:
        raise FileNotFoundError(f"在 {model_path} 中找不到 .safetensors 文件")

    print(f"  发现 {len(files)} 个 safetensors 文件")
    state_dict = {}
    for fpath in files:
        with safetensors.safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # 只保留 language model 的键（跳过 visual 和 mtp）
    # 权重在 safetensors 中前缀为 model.language_model.*，
    # 在我们的 ForCausalLM 中前缀为 model.*（因 self.model = TextModel）
    lm_prefix = "model.language_model."
    lm_state = {}
    for k, v in state_dict.items():
        if k.startswith(lm_prefix):
            # 替换前缀: model.language_model.xxx -> model.xxx
            new_key = "model" + k[len("model.language_model"):]
            lm_state[new_key] = v
        elif k.startswith("model.visual.") or k.startswith("mtp."):
            continue  # 跳过视觉和 MTP 权重

    print(f"  总权重: {len(state_dict)} 个张量")
    print(f"  语言模型: {len(lm_state)} 个张量")

    # 加载到模型
    result = model.load_state_dict(lm_state, strict=False)
    if result.missing_keys:
        # lm_head.weight 和 inv_freq buffer 是意料中的缺失
        print(f"  Missing keys（可接受）: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Warning: Unexpected keys: {result.unexpected_keys}")

    # lm_head.weight 与 embed_tokens.weight 绑定
    model.lm_head.weight = model.model.embed_tokens.weight
    print(f"  lm_head.weight 已绑定到 embed_tokens.weight（tie_word_embeddings）")

    # 移动到目标设备
    model.to(device=device, dtype=dtype)
    print(f"  模型已移至 {device}, dtype={dtype}")


# ============================================================================
# 自回归生成
# ============================================================================

@torch.no_grad()
def generate(
    model: Qwen3_5ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: int = 248044,
) -> tuple[torch.Tensor, dict]:
    """自回归生成（无 KV cache）"""
    generated = input_ids.clone()
    stats = {"step_times": [], "input_len": input_ids.shape[1]}

    for step in range(max_new_tokens):
        t0 = time.perf_counter()

        logits = model(generated)
        dt = time.perf_counter() - t0
        stats["step_times"].append(dt)

        next_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature

            if top_k > 0:
                topk_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values
                next_logits = next_logits.masked_fill(
                    next_logits < topk_vals[..., -1:], float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                remove = mask.scatter(-1, sorted_idx, mask)
                next_logits = next_logits.masked_fill(remove, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == eos_token_id:
            break

    return generated, stats


# ============================================================================
# 配置读取 — Qwen3.5 多模态嵌套 config
# ============================================================================

def load_text_config(model_path: str):
    """从 config.json 中提取 text_config 并返回简单对象"""
    with open(f"{model_path}/config.json", "r") as f:
        full_config = json.load(f)

    # text_config 可能在顶层（纯文本）或嵌套在 text_config 中（多模态）
    if "text_config" in full_config:
        tc_data = full_config["text_config"]
    else:
        tc_data = full_config

    # 从 rope_parameters 中提取
    rope_params = tc_data.get("rope_parameters", {})

    class TextConfig:
        pass

    cfg = TextConfig()
    cfg.hidden_size = tc_data["hidden_size"]
    cfg.num_hidden_layers = tc_data["num_hidden_layers"]
    cfg.num_attention_heads = tc_data["num_attention_heads"]
    cfg.num_key_value_heads = tc_data["num_key_value_heads"]
    cfg.head_dim = tc_data.get("head_dim",
                                cfg.hidden_size // cfg.num_attention_heads)
    cfg.intermediate_size = tc_data["intermediate_size"]
    cfg.vocab_size = tc_data["vocab_size"]
    cfg.rms_norm_eps = tc_data.get("rms_norm_eps", 1e-6)
    cfg.rope_theta = rope_params.get("rope_theta", 10000000.0)
    cfg.partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
    cfg.layer_types = tc_data["layer_types"]
    cfg.linear_num_key_heads = tc_data["linear_num_key_heads"]
    cfg.linear_num_value_heads = tc_data["linear_num_value_heads"]
    cfg.linear_key_head_dim = tc_data["linear_key_head_dim"]
    cfg.linear_value_head_dim = tc_data["linear_value_head_dim"]
    cfg.linear_conv_kernel_dim = tc_data["linear_conv_kernel_dim"]
    cfg.tie_word_embeddings = tc_data.get("tie_word_embeddings", True)
    cfg.max_position_embeddings = tc_data.get("max_position_embeddings", 262144)

    return cfg


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="从零手写 Qwen3.5-0.8B 推理")
    parser.add_argument("--model", type=str, required=True,
                        help="模型本地路径，如 D:\\code\\model\\Qwen3.5-0.8B")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备 (默认: CPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3.5-0.8B 从零手写 — 端到端推理")
    print("  Gated DeltaNet + GQA Full Attention 混合架构")
    print("=" * 60)

    # --- 设备 ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        best = free.index(max(free))
        device = torch.device(f"cuda:{best}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}, dtype: {dtype}")

    # --- 读取配置 ---
    print(f"\n[1/4] 读取模型配置...")
    config = load_text_config(args.model)
    print(f"  Model: {args.model}")
    print(f"  Layers: {config.num_hidden_layers} ({config.layer_types[0]}..)")
    print(f"  Hidden: {config.hidden_size}, Heads: {config.num_attention_heads}")
    print(f"  KV Heads: {config.num_key_value_heads}, Head dim: {config.head_dim}")
    print(f"  Rotary dim: {int(config.head_dim * config.partial_rotary_factor)}")

    # --- 创建模型 ---
    print(f"\n[2/4] 创建模型结构...")
    model = Qwen3_5ForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,} ({param_count / 1e9:.2f}B)")

    # --- 加载权重 ---
    print(f"\n[3/4] 从 safetensors 加载权重...")
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    # --- 生成 ---
    print(f"\n[4/4] 生成文本...")
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except ImportError:
        print("  Warning: transformers not available, using basic tokenizer")
        tokenizer = None

    if tokenizer is not None:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        eos_id = tokenizer.eos_token_id
    else:
        # 一个简单的占位 tokenizer
        print("  No tokenizer available, using dummy input")
        vocab_size = config.vocab_size
        dummy_ids = torch.randint(0, min(vocab_size, 1000), (1, 10)).to(device)
        input_ids = dummy_ids
        eos_id = 248044

    sampling = "greedy" if args.temperature == 0 else f"T={args.temperature}"
    print(f"  Prompt: {args.prompt}")
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Sampling: {sampling}, max_tokens: {args.max_tokens}")

    t0 = time.perf_counter()
    output_ids, stats = generate(
        model, input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )
    t_total = time.perf_counter() - t0

    new_tokens = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True) if tokenizer else str(new_tokens.tolist())

    print(f"\n{'=' * 60}")
    print("Generated:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    n = len(new_tokens)
    step_times = stats["step_times"]
    print(f"\n统计:")
    print(f"  生成 tokens: {n}")
    print(f"  总耗时: {t_total:.2f}s")
    if n > 0:
        print(f"  平均每步: {sum(step_times) / len(step_times) * 1000:.1f} ms")
        print(f"  首步 (prefill): {step_times[0] * 1000:.1f} ms")
        if len(step_times) > 1:
            print(f"  末步: {step_times[-1] * 1000:.1f} ms")


if __name__ == "__main__":
    main()

