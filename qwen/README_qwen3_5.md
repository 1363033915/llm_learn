# Lesson 2 扩展：从零手写 Qwen3.5-0.8B 推理

在标准 Qwen3（稠密 GQA）的基础上，Qwen3.5 引入了**混合架构**——交错排列 Gated DeltaNet（线性注意力）和标准 GQA 注意力层。本扩展用纯 PyTorch 从零实现 Qwen3.5-0.8B 的完整推理，仅依赖 `safetensors` 加载权重。

## 学习目标

1. 理解 Gated DeltaNet 的递推原理——用状态矩阵代替 KV cache 的线性注意力
2. 理解混合架构的设计动机：线性注意力层处理长序列高效 → 每 4 层插入一个 full attention 层做全局纠错
3. 掌握 Qwen3.5 的特有组件：Partial RoPE、Output Gate、Qwen3_5RMSNorm
4. 从本地 safetensors 文件加载权重，无需 `transformers.AutoModel`

## Qwen3 vs Qwen3.5-0.8B 架构对比

```
Qwen3-0.6B (全部稠密)                 Qwen3.5-0.8B (混合)

Layer 0:  GQA Full Attention          Layer 0:  GatedDeltaNet  ← 线性
Layer 1:  GQA Full Attention          Layer 1:  GatedDeltaNet  ← 线性
Layer 2:  GQA Full Attention          Layer 2:  GatedDeltaNet  ← 线性
Layer 3:  GQA Full Attention          Layer 3:  GQA Full Attention  ← 稠密 (每4层一个)
...                                    ...
                                     Layer 22: GatedDeltaNet
                                     Layer 23: GQA Full Attention
```

| 特性 | Qwen3-0.6B | Qwen3.5-0.8B |
|------|-----------|-------------|
| 层数 | 28 | 24 |
| 注意力类型 | 100% GQA | 75% GatedDeltaNet + 25% GQA |
| layers 属性 | 无分类 | `layer_types: ["linear_attention" \| "full_attention"]` |
| RoPE | 完整 head_dim | Partial (25%，64/256 dim) |
| Output Gate | 无 | 有 (`q_proj` 2× 输出，sigmoid 门控) |
| RMSNorm | `x * weight`（weight=1） | `x * (1 + weight)`（weight=0） |
| tie_word_embeddings | 通常 | True（lm_head 与 embed_tokens 共享权重） |

## 两大注意力模块详解

### Gated DeltaNet — 线性注意力

Gated DeltaNet 用一个固定大小的**状态矩阵** $S_t \in \mathbb{R}^{d_k \times d_v}$ 维护历史信息，每步仅做一次外积更新，计算量为 $O(d_k d_v)$ 而非 softmax 注意力的 $O(t^2)$。

**递推公式**：

$$
\begin{aligned}
S_t &= g_t \cdot S_{t-1} + k_t \otimes (\beta_t \odot (v_t - S_{t-1}^\top k_t)) \\
o_t &= S_t^\top q_t
\end{aligned}
$$

**数据流**：

```
hidden_states (1, S, 1024)
    │
    ├── in_proj_qkv → (1, S, 6144)
    │       ├── causal_conv1d (depthwise, kernel=4, groups=6144)
    │       └── split → q (1, S, 16, 128), k (1, S, 16, 128), v (1, S, 16, 128)
    │
    ├── in_proj_a → (1, S, 16) ─ + dt_bias ─ softplus ─ × -exp(A_log) → g (衰减因子)
    ├── in_proj_b → (1, S, 16) ─ sigmoid → beta (更新门控)
    │
    └── in_proj_z → (1, S, 16, 128) ─ silu ─ × norm(core_attn_out) → 门控输出

递推：
    for t in 1..S:
        q_t, k_t, v_t 先 L2 归一化
        state *= exp(g_t)                ← 历史衰减
        kv_mem = state @ k_t             ← 当前键的记忆
        delta = (v_t - kv_mem) * beta_t  ← 增量规则
        state += k_t ⊗ delta             ← 状态更新
        out[t] = state @ q_t             ← 输出读出
```

**关键组件**：

- **Causal Conv1d**（kernel=4）：在 QKV 投影后做深度可分离因果卷积，捕获局部依赖（类 Mamba）
- **L2 归一化 Q/K**：防止数值爆炸，使注意力更稳定
- **A_log / dt_bias**：可学习的逐头衰减参数（初始值为均匀分布的对数）
- **beta (sigmoid(b))**：可学习的更新门控，控制新信息进入状态的程度

### Full Attention — 带 Output Gate 的 GQA

与标准 Qwen3 的注意力类似，但有 3 个区别：

| 特性 | 标准 Qwen3 | Qwen3.5 Full Attention |
|------|-----------|----------------------|
| RoPE 范围 | 全部 head_dim | 仅前 25%（64/256 dim），后 75% 不旋转 |
| q_proj 输出 | `heads × head_dim` | `heads × head_dim × 2`（多出一倍做 output gate） |
| RMSNorm 实现 | `x * weight` | `x * (1 + weight)` |
| Output Gate | 无 | `attn_out * sigmoid(gate)` |

**Qwen3_5RMSNorm 的两版实现**：

```python
# 标准版（基础组件）
class RMSNorm:
    weight = init_ones()      # 初始化为 1
    def forward(x):
        return x / RMS(x) * weight       # 直接乘 weight

# Qwen3.5 版（注意差异！）
class Qwen3_5RMSNorm:
    weight = init_zeros()     # 初始化为 0！训练后非零
    def forward(x):
        return x / RMS(x) * (1 + weight)  # 乘 (1 + weight)
```

> 初始化时 weight=0 → 等价于直接归一化，训练过程中 weight 逐步学习偏移。这种实现让初始行为更接近恒等映射。

## Qwen3.5 多模态 config 嵌套结构

`Qwen3.5-0.8B` 是多模态模型（虽然实际只用了文本解码器），`config.json` 是嵌套的：

```json
{
    "model_type": "qwen3_5",
    "text_config": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "layer_types": ["linear_attention", ..., "full_attention"],
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25
        }
    },
    "vision_config": { ... }  ← 本脚本不加载
}
```

权重在 safetensors 中也有对应嵌套前缀：
```
model.language_model.embed_tokens.weight   ← 文本 embedding
model.language_model.layers.0.lin�ar_attn.… ← 线性注意力层
model.language_model.layers.3.self_attn.…  ← 稠密注意力层
model.language_model.norm.weight           ← Final RMSNorm
model.visual.blocks.0.…                    ← 视觉编码器（跳过）
mtp.fc.weight                               ← 多 token 预测（跳过）
```

## 快速开始

```bash
# 依赖
pip install torch safetensors transformers

# 贪心解码（temperature=0，输出确定）
python resources/lesson-2-run-qwen3/run_qwen3_5_0_8B.py \
    --model "D:\code\model\Qwen3.5-0.8B" \
    --temperature 0 \
    --prompt "What is the capital of France?"

# 采样生成
python resources/lesson-2-run-qwen3/run_qwen3_5_0_8B.py \
    --model "D:\code\model\Qwen3.5-0.8B" \
    --temperature 0.7 \
    --prompt "Explain AI in one sentence." \
    --max-tokens 64
```

## 验证结果

与 HuggingFace `AutoModelForCausalLM` 对比（2026-06-26，transformers 5.12.1）：

```
Max logit difference (full): 0.0000   ← 精确匹配
Max logit difference (last): 0.0000
Top-5 match: True                     ← "The", "Paris", "There", "**", "Capital"
Argmax match: True                    ← 都预测 "The"
```

## 文件结构

```
resources/lesson-2-run-qwen3/
├── run_qwen3.py             # 从零手写标准 Qwen3（稠密 GQA，~650 行）
├── run_qwen3_5_0_8B.py     # 从零手写 Qwen3.5-0.8B（混合架构，~790 行）
├── README.md                # 标准 Qwen3 课程文档
├── README_qwen3_5.md        # 本文档（Qwen3.5 混合架构）
├── image.png                # 生成过程示意图
└── requirements.txt
```

## 性能说明

当前没有 KV cache 和 GatedDeltaNet 的状态缓存，每步重新计算整个序列。CPU 上约 1-4 秒/步（随序列长度线性增长）。后续课程将实现状态缓存，使 decode 步骤从 O(seq_len × d_k d_v) 降为 O(d_k d_v)。

```
Step 0: forward(prompt)               — pref�ll，处理 N 个 token
Step 1: forward(prompt + tok1)         — 处理 N+1 个 token
Step 2: forward(prompt + tok1,2)       — 处理 N+2 个 token
...
Step T: forward(prompt + tok1..T)      — 越来越慢

线性注意力（GatedDeltaNet）：O(T × d_k d_v) ≈ constant per step
稠密注意力（Full Attention）：O(T²) per step（本文无 KV cache）
```

## 练习

1. **逐模块理解**：打印 GatedDeltaNet 每个投影的输出形状，画出数据流图。

2. **对比标准 Qwen3**：用 `run_qwen3.py` 跑一个稠密模型，对比首步 prefill 时间和后续 decode 时间的增长曲线。

3. **验证权重加载**：用 `temperature=0` 对比我们的实现和 HF 的完整 logits，验证每个位置都精确匹配。

4. **分析 A_log**：打印每层 GatedDeltaNet 的 `A_log` 值（`-log(exp(A_log).mean())`），分析不同层的衰减速率差异。

## 参考资料

- [Qwen3.5 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [Gated DeltaNet 论文](https://arxiv.org/abs/2412.06464) — Hybrid Linear Attention for Long-Context Language Models
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — State Space Models
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding
- [GQA 论文](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
- [HuggingFace modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
