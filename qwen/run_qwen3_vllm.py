"""
Lesson 2 扩展: 用 vLLM 引擎推理 Qwen3

对照 run_qwen3_update.py（手写 PyTorch ~400 行），看看 vLLM 引擎如何在 ~80 行内
完成同样的任务——同时自动获得：
  - PagedAttention KV cache（O(1) decode）
  - 多批次并行推理
  - Prefix caching
  - Tensor parallelism

用法:
    python run_qwen3_vllm.py --model Qwen/Qwen3-0.6B
    python run_qwen3_vllm.py --model Qwen/Qwen3-8B \
        --prompt "Tell me about AI" --temperature 0 --max-tokens 256
"""

import time
import argparse

import torch


# ============================================================================
# 1. 手写 vs vLLM 对照表
# ============================================================================
#
# ┌────────────────────────────┬─────────────────────────────────┐
# │ 手写 PyTorch (run_qwen3.py)│ vLLM (本文)                     │
# ├────────────────────────────┼─────────────────────────────────┤
# │ RMSNorm class              │ vLLM 内置实现 (融合 CUDA kernel) │
# │ RotaryEmbedding class      │ vLLM 内置 (RoPE + interleaved)   │
# │ Qwen3Attention class       │ PagedAttention + FlashInfer      │
# │ GQA + repeat_kv            │ 自动处理                         │
# │ SwiGLU MLP                 │ 内置                             │
# │ 手写 causal_mask           │ vLLM 自动构建                    │
# │ load_weights_from_hf()     │ vLLM 加载 + 分片 + TP            │
# │ generate() 自回归循环       │ LLM.generate() / AsyncLLM        │
# │ 无 KV cache（O(N²)）       │ PagedAttention（O(N)）           │
# │ 单序列                      │ 批量并行                        │
# └────────────────────────────┴─────────────────────────────────┘


# ============================================================================
# 2. vLLM 推理
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="vLLM 推理 Qwen3")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallelism size (多 GPU)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU 显存利用率")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="最大上下文长度 (不指定则用 config 里的值)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="禁用 CUDA graph (调试用)")
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM 引擎推理 Qwen3")
    print("=" * 60)

    # --- 检查可用 GPU ---
    gpu_count = torch.cuda.device_count()
    device_info = f"GPU × {gpu_count}" if gpu_count > 0 else "CPU (fallback)"
    print(f"\nDevice: {device_info}")
    if gpu_count == 0:
        print("⚠ 警告: 未检测到 GPU, vLLM 仅支持 CUDA 推理")

    # --- 加载模型 ---
    print(f"\n[1/3] 加载 vLLM 引擎...")
    print(f"  Model: {args.model}")

    # 延迟导入，避免非 GPU 环境直接报错
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    t0 = time.perf_counter()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        # 使用 bfloat16（GPU 高效），非 GPU 环境会自动处理
        dtype="bfloat16" if gpu_count > 0 else "auto",
    )

    dt_load = time.perf_counter() - t0
    print(f"  加载耗时: {dt_load:.1f}s")

    # --- 设置采样参数 ---
    print(f"\n[2/3] 设置采样参数...")

    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0.0,
        top_k=args.top_k if args.temperature > 0 else -1,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
        # 当 temperature=0 时 vLLM 自动切换贪心解码
    )

    sampling_desc = "greedy" if args.temperature == 0 else f"T={args.temperature}"
    print(f"  Sampling: {sampling_desc}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Top-K: {args.top_k}, Top-P: {args.top_p}")

    # --- 构建输入 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(text)

    print(f"  Prompt: {args.prompt}")
    print(f"  Input tokens: {len(prompt_tokens)}")

    # --- 生成 ---
    print(f"\n[3/3] 推理中...")

    t_gen_start = time.perf_counter()
    outputs = llm.generate(text, sampling_params)
    t_gen = time.perf_counter() - t_gen_start

    # --- 解析结果 ---
    result = outputs[0]
    output_text = result.outputs[0].text
    output_tokens = result.outputs[0].token_ids

    # 获取 prompt 的 token ID 用于统计
    prompt_token_ids = result.prompt_token_ids

    print(f"\n{'=' * 60}")
    print("Generated:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # --- 统计 ---
    n_prompt = len(prompt_token_ids)
    n_output = len(output_tokens)

    print(f"\n统计:")
    print(f"  Prompt tokens: {n_prompt}")
    print(f"  Output tokens: {n_output}")
    print(f"  推理耗时: {t_gen:.2f}s")
    if n_output > 0:
        # vLLM Metrics（如果可用）
        metrics = result.metrics
        if metrics:
            ttft = getattr(metrics, 'time_to_first_token', None)
            tpot = getattr(metrics, 'time_per_output_token', None)
            if ttft is not None:
                print(f"  TTFT (首 token 延迟): {ttft * 1000:.1f} ms")
            if tpot is not None:
                print(f"  TPOT (每 token 平均): {tpot * 1000:.1f} ms")
            if n_output > 1 and ttft is not None:
                decode_time = t_gen - ttft
                decode_tokens = n_output - 1
                if decode_tokens > 0:
                    print(f"  Decode 阶段: {decode_time:.2f}s / {decode_tokens} tokens = "
                          f"{decode_time / decode_tokens * 1000:.1f} ms/token")
            # 吞吐量
            if t_gen > 0:
                print(f"  吞吐量: {n_output / t_gen:.1f} tokens/s")
        else:
            print(f"  平均: {t_gen / max(n_output, 1) * 1000:.1f} ms/token")

    print(f"\n对比手写版本 (run_qwen3_update.py):")
    print(f"  手写版: 每步重新计算整个序列 → O(N²) 注意力")
    print(f"  vLLM:   PagedAttention KV cache → O(N) 解码, 批量并行")
    print(f"  手写版: 需手写 ~400 行 (RMSNorm/RoPE/GQA/生成循环)")
    print(f"  vLLM:   引擎封装 ~80 行 (只配参数 + 写 prompt)")
    print(f"  手写版: 无 KV cache, decode 渐进加速为 0")
    print(f"  vLLM:   PagedAttention, 恒定 decode 速度")


if __name__ == "__main__":
    main()
