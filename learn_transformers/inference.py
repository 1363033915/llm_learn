# -*- coding: utf-8 -*-
"""
Sample from a trained model
"""
import torch
import tiktoken
from Model import Model

# 加载模型和超参数
checkpoint = torch.load('model/model.ckpt')
h_params = checkpoint['h_params']
model = Model(h_params)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(h_params['device'])

# 加载分词器
encoding = tiktoken.get_encoding("cl100k_base")

# 输入 prompt
start = "酸奶"
start_ids = encoding.encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=h_params['device'])[None, ...]

# 生成
with torch.no_grad():
    y = model.generate(x, max_new_tokens=200, temperature=1, top_k=None)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model param size: {total_params:,}")

# 打印模型参数
for name in model.state_dict().keys():
    print(name, model.state_dict()[name].shape)