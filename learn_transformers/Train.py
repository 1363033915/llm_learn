"""
Train a Transformer model
"""
import os
import torch
import tiktoken
from Model import Model

# GPU 内存配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()

# 超参数
h_params = {
    "d_model": 80,
    "batch_size": 2,
    "context_length": 128,
    "num_blocks": 6,
    "num_heads": 4,
    "dropout": 0.1,
    "max_iters": 500,
    "learning_rate": 1e-3,
    "eval_interval": 50,
    "eval_iters": 10,
    "device": "cuda" if torch.cuda.is_available() else
              ("mps" if torch.backends.mps.is_available() else "cpu"),
    "TORCH_SEED": 1337
}
torch.manual_seed(h_params["TORCH_SEED"])

# 加载数据
with open('data/GoodsOrder.csv', 'r', encoding='gbk') as file:
    text = file.read()

# 分词
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)
max_token_value = max(tokenized_text) + 1
h_params['max_token_value'] = max_token_value
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=h_params['device'])

print(f"Total: {len(tokenized_text):,} tokens")

# 分割数据
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# 初始化模型
model = Model(h_params).to(h_params['device'])


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - h_params['context_length'],
                         size=(h_params['batch_size'],))
    x = torch.stack([data[idx:idx + h_params['context_length']] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + h_params['context_length'] + 1] for idx in idxs])
    return x.to(h_params['device']), y.to(h_params['device'])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(h_params['eval_iters'])
        for k in range(h_params['eval_iters']):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=h_params['learning_rate'])

for step in range(h_params['max_iters']):
    if step % h_params['eval_interval'] == 0 or step == h_params['max_iters'] - 1:
        losses = estimate_loss()
        print(f'Step: {step}, Training Loss: {losses["train"]:.3f}, '
              f'Validation Loss: {losses["valid"]:.3f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 保存模型
if not os.path.exists('model/'):
    os.makedirs('model/')

torch.save({
    'model_state_dict': model.state_dict(),
    'h_params': h_params
}, 'model/model.ckpt')

print("Training complete. Model saved to model/model.ckpt")
