import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 导入你在 train.py 中定义的自定义模型类
# 确保 train.py 和 llm_test.py 在同一个目录下
from train import LLM, Config

# ==========================================
# 1. 注册自定义模型 (保持与你训练时一致)
# ==========================================
# 这一步是为了让 transformers 库认识你的 "small_model"
AutoConfig.register("small_model", Config)
AutoModelForCausalLM.register(Config, LLM)

# ==========================================
# 2. 加载模型和分词器
# ==========================================
model_path = './saves/model'
print(f"🚀 正在从 {model_path} 加载模型...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # 切换到评估模式
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("💡 提示：请确保 ./saves/model 文件夹存在且包含 config.json 和 pytorch_model.bin")
    exit()


# ==========================================
# 3. 定义推理函数
# ==========================================
def chat(instruction):
    # --- A. 构造输入 (必须与训练时的 DataSet 格式完全一致) ---
    # 训练时格式: <s>### Instruction:\n{instruction}\n\n### Response:\n{output}</s>
    prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # 编码
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # --- B. 生成 (处理生成器返回值) ---
    print(f"\n🤖 正在思考: {instruction} ...")

    all_tokens = []

    # 注意：这里调用的是你自定义 model 的 generate 方法
    # 根据你的报错，它返回的是一个 generator (生成器)
    # 参数说明:
    # - 第一个参数是 inputs (字典格式，包含 input_ids)
    # - eos_token_id: 结束符
    # - max_len: 最大生成长度 (根据你的代码推测是第3个参数)
    # - stream=False: 强制关闭流式，虽然你代码里还是用了 for 循环来收集
    try:
        for token_batch in model.generate(
                {"input_ids": input_ids, "labels": None},
                tokenizer.eos_token_id,
                200,  # 最大长度，防止生成太长
                stream=False,  # 你的代码逻辑似乎支持这个参数
                temperature=0.7,  # 稍微增加一点随机性，避免死板
                top_k=50  # 增大 top_k，避免只选标点符号
        ):
            # 收集生成的 token (假设 token_batch 是 tensor 格式)
            # 这里假设每次 yield 出来的是 (batch_size, 1) 的 tensor
            if isinstance(token_batch, torch.Tensor):
                all_tokens.append(token_batch[0])  # 取第0个batch
            else:
                # 如果是列表或其他格式，请根据实际情况调整
                all_tokens.append(torch.tensor([token_batch]))

    except Exception as e:
        print(f"❌ 生成出错: {e}")
        return

    # --- C. 解码与后处理 ---
    if len(all_tokens) > 0:
        # 拼接所有 tokens
        full_sequence = torch.cat(all_tokens, dim=0)

        # 解码 (跳过特殊符号)
        generated_text = tokenizer.decode(full_sequence, skip_special_tokens=True)

        # 去除 Prompt 部分，只保留回答
        # 因为生成的内容包含了输入的 prompt，我们需要切分它
        if "### Response:\n" in generated_text:
            response = generated_text.split("### Response:\n")[-1]
            print(f"💬 回答: {response.strip()}")
        else:
            print(f"💬 完整输出: {generated_text}")
    else:
        print("💬 模型未生成任何内容 (可能是 EOS 触发过早)")


# ==========================================
# 4. 开始测试
# ==========================================
if __name__ == "__main__":
    print("--- 模型已就绪，输入 'quit' 退出 ---")
    while True:
        user_input = input("\n请输入指令: ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip():
            continue

        chat(user_input)