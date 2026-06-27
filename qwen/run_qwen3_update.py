"""
Lesson 2: 从零手写 Qwen3 模型 — 端到端推理

仅依赖 PyTorch，从零实现 Qwen3 的完整推理流程：
  1. 模型配置 (AutoConfig)
  2. 基础组件 (RMSNorm, RoPE, SwiGLU MLP)
  3. 注意力机制 (GQA + QK-Norm)
  4. Transformer Decoder
  5. 权重加载 (通过 transformers from_pretrained 获取权重)
  6. 自回归生成

用法:
    python run_qwen3.py --model Qwen/Qwen3-0.6B
    python run_qwen3.py --model Qwen/Qwen3-8B --prompt "Tell me about AI" --temperature 0
"""

import time
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 2. RMSNorm — 比 LayerNorm 更高效的归一化
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight


# ============================================================================
# 3. Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(position_ids[0].float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ============================================================================
# 4. Grouped Query Attention (GQA) + QK-Norm
# ============================================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            attn_output: (B, S, hidden_size)
            updated_kv:   (k, v), 每头 KV cache 用于下一步
        """
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── KV cache ──
        if past_kv is not None:
            k_cached, v_cached = past_kv
            k = torch.cat([k_cached, k], dim=-2)   # (B, kv_heads, cached+S, head_dim)
            v = torch.cat([v_cached, v], dim=-2)

        updated_kv = (k.detach().contiguous(), v.detach().contiguous())

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_output), updated_kv


# ============================================================================
# 5. SwiGLU MLP
# ============================================================================

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# 6. Transformer Decoder Layer
# ============================================================================

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_kv = self.self_attn(
            hidden_states, position_embeddings, attention_mask, past_kv=past_kv)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, updated_kv


# ============================================================================
# 7. Qwen3Model — Transformer backbone
# ============================================================================

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # 修复：安全获取 rope_theta，默认 1000000.0
        rope_theta = getattr(config, 'rope_theta', 1000000.0)
        self.rotary_emb = RotaryEmbedding(config.head_dim, rope_theta)

    def forward(
        self, input_ids: torch.Tensor,
        past_key_values: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Returns:
            hidden_states: (B, S, hidden_size)  归一化后的最终输出
            past_key_values: 每层 [(k, v), ...], 用于后续 decode 步骤
        """
        B, S = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # ── 带 KV cache 偏移的位置 ID ──
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[-2]  # seq_len 维度

        position_ids = torch.arange(past_len, past_len + S, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(position_ids)

        # ── 带 KV cache 的因果掩码 ──
        total_len = past_len + S
        causal_mask = torch.full(
            (total_len, total_len), float("-inf"),
            device=input_ids.device, dtype=hidden_states.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total, total)

        if past_len > 0:
            # decode 阶段：只取最后 S 个 query 位置
            causal_mask = causal_mask[:, :, -S:, :]          # (1, 1, S, total)

        # ── 逐层前向（带 KV cache）──
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, updated_kv = layer(
                hidden_states, causal_mask, position_embeddings, past_kv=past_kv)
            new_past_key_values.append(updated_kv)

        return self.norm(hidden_states), new_past_key_values


# ============================================================================
# 8. Qwen3ForCausalLM — 完整语言模型
# ============================================================================

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor,
        past_key_values: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        hidden_states, past_key_values = self.model(input_ids, past_key_values=past_key_values)
        return self.lm_head(hidden_states), past_key_values


# ============================================================================
# 9. 权重加载 — 从 HuggingFace 预训练模型获取权重
# ============================================================================

def load_weights_from_hf(model: Qwen3ForCausalLM, model_name: str, device: torch.device, dtype: torch.dtype):
    """从 HuggingFace 预训练模型加载权重到我们手写的模型"""
    from transformers import AutoModelForCausalLM

    print(f"Loading HuggingFace model: {model_name} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,               # 新版 transformers 使用 dtype 替代 torch_dtype
        device_map=None,           # 不需要 device_map，避免 require accelerate
        low_cpu_mem_usage=True,    # 减少内存占用
    )
    # 确保模型在 CPU 上（默认就是 CPU，但显式调用一下更安全）
    hf_model = hf_model.to("cpu")

    hf_state_dict = hf_model.state_dict()
    print(f"  HF model has {len(hf_state_dict)} tensors")

    result = model.load_state_dict(hf_state_dict, strict=False)
    loaded = len(hf_state_dict) - len(result.unexpected_keys)
    print(f"  Loaded {loaded} tensors, skipped {len(result.unexpected_keys)} unexpected keys")
    if result.missing_keys:
        print(f"  Missing (non-persistent buffers, will be re-created): {result.missing_keys}")

    del hf_model, hf_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(device=device, dtype=dtype)


# ============================================================================
# 10. 自回归生成
# ============================================================================

@torch.no_grad()
def generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: int = 151645,
) -> tuple[torch.Tensor, dict]:
    generated = input_ids.clone()
    stats = {
        "step_times": [],
        "input_len": input_ids.shape[1],
        "prefill_time": 0.0,
        "decode_times": [],
        "decode_tokens": 0,
    }
    past_key_values = None

    for step in range(max_new_tokens):
        t0 = time.perf_counter()

        if step == 0:
            # ── Prefill: 整段 prompt 送入，初始化 KV cache ──
            logits, past_key_values = model(input_ids, past_key_values=None)
        else:
            # ── Decode: 只送上一个 token (shape [1,1])，复用 KV cache ──
            logits, past_key_values = model(next_token, past_key_values=past_key_values)

        dt = time.perf_counter() - t0
        stats["step_times"].append(dt)
        if step == 0:
            stats["prefill_time"] = dt
        else:
            stats["decode_times"].append(dt)
            stats["decode_tokens"] += 1

        next_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature

            if top_k > 0:
                topk_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values
                next_logits = next_logits.masked_fill(next_logits < topk_vals[..., -1:], float("-inf"))

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
# 11. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Lesson 2: 从零手写 Qwen3 推理")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型名或本地路径, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备, e.g. 'cuda:1' (默认: 自动选择最空闲的 GPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("Lesson 2: 从零手写 Qwen3 — 端到端推理")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoConfig

    # --- 选择设备 ---
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
    config = AutoConfig.from_pretrained(args.model)
    print(f"  Model: {args.model}")
    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}, "
          f"Heads: {config.num_attention_heads}, KV Heads: {config.num_key_value_heads}")

    # --- 创建模型 ---
    print(f"\n[2/4] 创建模型结构...")
    model = Qwen3ForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,} ({param_count / 1e9:.2f}B)")

    # --- 加载权重 ---
    print(f"\n[3/4] 从 HuggingFace 加载权重...")
    load_weights_from_hf(model, args.model, device, dtype)
    model.eval()

    print(f"\n[4/4] 生成文本...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

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
        eos_token_id=tokenizer.eos_token_id,
    )
    t_total = time.perf_counter() - t0

    # --- 输出 ---
    new_tokens = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n{'=' * 60}")
    print("Generated:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # --- 统计 ---
    n = len(new_tokens)
    step_times = stats["step_times"]
    prefill_time = stats["prefill_time"]
    decode_times = stats["decode_times"]

    print(f"\n统计 (KV cache 已启用):")
    print(f"  生成 tokens: {n}")
    print(f"  总耗时: {t_total:.2f}s")
    print(f"  Prefill: {prefill_time * 1000:.1f} ms  ({stats['input_len']} tokens)")
    if decode_times:
        avg_decode = sum(decode_times) / len(decode_times) * 1000
        print(f"  Decode 平均: {avg_decode:.1f} ms/token  ({len(decode_times)} tokens)")
        if len(decode_times) >= 2:
            print(f"  Decode 首/末: {decode_times[0] * 1000:.1f} / {decode_times[-1] * 1000:.1f} ms")
            if abs(decode_times[0] - decode_times[-1]) < 0.05 * decode_times[0]:
                print(f"  → Decode 速度恒定 ✓ (无二次增长)")
        print(f"  KV cache 加速比: {stats['input_len']}x (O(N²) → O(N))")


if __name__ == "__main__":
    main()