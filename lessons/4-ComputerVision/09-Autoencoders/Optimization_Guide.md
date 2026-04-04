# PyTorch 深度学习优化技巧指南

本文档总结了在训练自编码器模型时使用的各种优化技巧和加速方法。

---

## 一、数据加载优化

### 1.1 DataLoader 优化参数

```python
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,           # 多进程加载，通常设为 CPU 核心数
    pin_memory=True,         # 锁页内存，加速 GPU 传输
    prefetch_factor=4,       # 预取因子，每个 worker 预取的 batch 数
    persistent_workers=True, # 持久化 workers，避免重复创建进程
    drop_last=True           # 丢弃最后一个不完整的 batch
)
```

**参数说明：**

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `num_workers` | 数据加载进程数 | 4-8（CPU密集型任务） |
| `pin_memory` | 使用锁页内存 | `True`（GPU训练时） |
| `prefetch_factor` | 每个 worker 预取的 batch 数 | 2-4 |
| `persistent_workers` | 保持 worker 进程存活 | `True`（当 num_workers > 0） |

### 1.2 自动批次大小调整

根据 GPU 内存自动调整批次大小：

```python
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_mem >= 20:      # RTX 3090, A100 等
        batch_size = 16384
    elif gpu_mem >= 10:    # RTX 3080, V100 等
        batch_size = 8192
    else:                  # RTX 2080, T4 等
        batch_size = 4096
else:
    batch_size = 256
```

---

## 二、混合精度训练 (AMP)

### 2.1 基本用法

混合精度训练使用 FP16 进行前向和反向传播，FP32 进行梯度更新，可以：
- 减少显存占用 50%+
- 加速训练 2-3 倍
- 保持模型精度

```python
from torch.amp import GradScaler, autocast

# 创建 GradScaler
scaler = GradScaler('cuda', enabled=True)

# 训练循环
for batch in dataloader:
    # 前向传播（自动混合精度）
    with autocast('cuda', enabled=True):
        output = model(input)
        loss = loss_fn(output, target)
    
    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2.2 梯度裁剪 + AMP

```python
with autocast('cuda', enabled=True):
    output = model(input)
    loss = loss_fn(output, target)

optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 在裁剪前 unscale
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 2.3 BCEWithLogitsLoss 与 AMP

**重要**：`BCELoss` 不支持 autocast！必须使用 `BCEWithLogitsLoss`。

```python
# 错误：BCELoss 不支持 autocast
loss_fn = nn.BCELoss()  # 会报错

# 正确：使用 BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()

# 模型输出层不要加 sigmoid
class Decoder(nn.Module):
    def forward(self, x):
        # ... 卷积操作 ...
        return x  # 输出 logits，不加 sigmoid

# 推理时手动加 sigmoid
output = torch.sigmoid(model(input))
```

---

## 三、网络结构优化

### 3.1 BatchNorm

在每个卷积层/全连接层后添加 BatchNorm：
- 加速收敛
- 稳定训练
- 允许更大的学习率

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 添加 BatchNorm
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # 在激活函数前
        x = F.relu(x)
        return x
```

### 3.2 Dropout 防止过拟合

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)  # 卷积层用 Dropout2d
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # 在激活函数后
        return x
```

### 3.3 Decoder 尺寸精确控制

使用 `F.interpolate` 替代转置卷积可以精确控制输出尺寸：

```python
class Decoder(nn.Module):
    def forward(self, x):
        # 精确控制尺寸：4x4 -> 7x7 -> 14x14 -> 28x28
        x = F.interpolate(x, size=7, mode='nearest')
        x = self.conv1(x)
        
        x = F.interpolate(x, size=14, mode='nearest')
        x = self.conv2(x)
        
        x = F.interpolate(x, size=28, mode='nearest')
        x = self.conv3(x)
        
        return x
```

---

## 四、训练优化技巧

### 4.1 学习率调度器

```python
from torch import optim

# ReduceLROnPlateau：验证损失停止下降时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控指标越小越好
    factor=0.5,      # 每次降低 50%
    patience=3       # 连续 3 个 epoch 没改善就降低
)

# 每个 epoch 结束后
scheduler.step(test_loss)
```

### 4.2 早停机制 (Early Stopping)

```python
best_test_loss = float('inf')
patience_counter = 0
early_stop_patience = 7

for epoch in range(epochs):
    # ... 训练和验证 ...
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        # 可选：保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= early_stop_patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
```

### 4.3 梯度裁剪

防止梯度爆炸，稳定训练：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4.4 权重衰减 (L2 正则化)

```python
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
```

### 4.5 高效梯度清零

```python
optimizer.zero_grad(set_to_none=True)  # 比 .zero_grad() 更高效
```

---

## 五、GPU 加速技巧

### 5.1 TF32 加速 (Ampere 架构 GPU)

```python
# 启用 TF32 加速矩阵乘法和卷积
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 5.2 cuDNN 自动优化

```python
if torch.cuda.is_available():
    # 为固定输入尺寸优化卷积算法
    torch.backends.cudnn.benchmark = True
    # 允许非确定性算法（更快）
    torch.backends.cudnn.deterministic = False
```

### 5.3 非阻塞数据传输

```python
# 数据传输到 GPU 时使用非阻塞模式
imgs = imgs.to(device, non_blocking=True)
```

### 5.4 显存监控

```python
if device.type == 'cuda':
    mem_used = torch.cuda.max_memory_allocated() / 1024**2
    print(f'GPU Memory: {mem_used:.0f}MB')
    torch.cuda.reset_peak_memory_stats()
```

---

## 六、VAE 专项优化

### 6.1 重参数化技巧

```python
# 数值稳定的重参数化
z_val = z_mean + torch.exp(0.5 * z_log) * eps  # 而不是 z_mean + z_std * eps
```

### 6.2 KL 散度计算

```python
# 数值稳定的 KL 散度
kl_loss = -0.5 * torch.sum(1 + z_log - z_mean.pow(2) - z_log.exp(), dim=1)
kl_loss = torch.mean(kl_loss)
```

---

## 七、AAE 专项优化

### 7.1 判别器使用 LeakyReLU

```python
class AAEDiscriminator(nn.Module):
    def __init__(self):
        self.relu = nn.LeakyReLU(0.2)  # 比 ReLU 更稳定
```

### 7.2 判别器输出 logits

判别器输出 logits 而非概率，使用 `BCEWithLogitsLoss`：

```python
class AAEDiscriminator(nn.Module):
    def forward(self, x):
        # ... 线性层 ...
        return self.linear5(x)  # 输出 logits，不加 sigmoid

# 训练时使用 BCEWithLogitsLoss
disc_loss = F.binary_cross_entropy_with_logits(
    disc_real, torch.ones_like(disc_real)
) + F.binary_cross_entropy_with_logits(
    disc_fake, torch.zeros_like(disc_fake)
)
```

---

## 八、PyTorch 新版 API 注意事项

### 8.1 AMP API 更新

PyTorch 2.0+ 推荐使用新版 API：

```python
# 旧版（已弃用）
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler(enabled=use_amp)
with autocast(enabled=use_amp):

# 新版（推荐）
from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda', enabled=use_amp)
with autocast('cuda', enabled=use_amp):
```

### 8.2 学习率调度器 verbose 参数

```python
# 旧版（已弃用）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', verbose=True
)

# 新版：使用 get_last_lr() 获取学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min'
)
current_lr = optimizer.param_groups[0]['lr']
```

---

## 九、性能对比

| 优化技巧 | 显存减少 | 训练加速 | 实现难度 |
|---------|---------|---------|---------|
| 混合精度训练 | 50%+ | 2-3x | 低 |
| DataLoader 优化 | - | 1.5-2x | 低 |
| BatchNorm | - | 1.2x | 低 |
| TF32 加速 | - | 1.1x | 极低 |
| 梯度裁剪 | - | 稳定训练 | 极低 |
| 早停机制 | - | 节省时间 | 低 |

---

## 十、完整训练模板

```python
def train(dataloaders, model, loss_fn, optimizer, epochs, device, use_amp=True):
    # AMP
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 早停
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        
        for batch in train_dataloader:
            imgs, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            
            # 前向传播（混合精度）
            with autocast('cuda', enabled=use_amp):
                preds = model(imgs)
                loss = loss_fn(preds, imgs)
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        test_loss = evaluate(model, test_dataloader, device)
        
        # 学习率调度
        scheduler.step(test_loss)
        
        # 早停检查
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                break
    
    return best_test_loss
```

---

## 参考资源

- [PyTorch AMP 官方文档](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
