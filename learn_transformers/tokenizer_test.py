# 代码示例
import tiktoken

# 使用 GPT-4 的编码器
enc = tiktoken.get_encoding("cl100k_base")

# 编码
# text = "小沈阳江西演唱会邀请了"
# tokens = enc.encode(text)
# print(f"Token IDs: {tokens}")
# print(f"Token 数量: {len(tokens)}")
#
# # 解码（把 Token ID 转回文字）
# decoded = enc.decode(tokens)
# print(f"解码结果: {decoded}")
#
# # 查看每个 Token 对应的文字
# for token_id in tokens:
#     print(f"  {token_id} → '{enc.decode([token_id])}'")


# # 代码示例
# import torch
# import torch.nn as nn
#
# # 创建 LayerNorm 层
# layer_norm = nn.LayerNorm(normalized_shape=4, bias=True)
#
# # 输入数据
# x = torch.tensor([[22.0, 5.0, 6.0, 8.0]])
#
# # 应用 LayerNorm
# y = layer_norm(x)
# print(y)  # 输出接近 [1.71, -0.76, -0.62, -0.33]

# 代码示例
import torch
import torch.nn.functional as F

# 输入数据（logits）
logits = torch.tensor([3.01, 0.09, 2.48, 1.95])

# 应用 Softmax
probs = F.softmax(logits, dim=0)
print(probs)  # tensor([0.5028, 0.0271, 0.2959, 0.1742])
print(probs.sum())  # tensor(1.0000)