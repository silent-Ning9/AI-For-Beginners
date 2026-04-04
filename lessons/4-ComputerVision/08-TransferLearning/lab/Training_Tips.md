# 深度学习模型训练技巧总结

本文档总结了在Pet Faces图像分类任务中使用的训练优化技巧。

## 1. 数据增强 (Data Augmentation)

数据增强是防止过拟合的最有效方法之一，通过对训练图像进行随机变换来增加数据多样性。

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),           # 调整大小
    transforms.RandomCrop(224),              # 随机裁剪
    transforms.RandomHorizontalFlip(),       # 随机水平翻转
    transforms.RandomRotation(15),           # 随机旋转（±15度）
    transforms.ColorJitter(                  # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(                    # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**作用**：
- 增加训练数据多样性
- 提高模型泛化能力
- 减少过拟合

## 2. 数据标准化 (Normalization)

使用ImageNet的均值和标准差进行标准化：

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

**作用**：
- 使输入数据分布一致
- 加速模型收敛
- 特别重要：使用预训练模型时必须使用相同的标准化参数

## 3. 模型架构优化

### 3.1 批归一化 (Batch Normalization)

在每个卷积层后添加BatchNorm：

```python
nn.Conv2d(3, 64, kernel_size=3, padding=1),
nn.BatchNorm2d(64),  # 批归一化
nn.ReLU(),
```

**作用**：
- 加速训练收敛
- 允许使用更高的学习率
- 有一定的正则化效果

### 3.2 全局平均池化 (Global Average Pooling)

用GAP替代大型全连接层：

```python
# 传统方法：大量参数
nn.Linear(256 * 14 * 14, 512)  # 50,176个输入

# 改进方法：GAP
nn.AdaptiveAvgPool2d((1, 1))   # 输出 512 x 1 x 1
nn.Linear(512, 256)            # 只有512个输入
```

**作用**：
- 大幅减少参数量（从26M降至5M）
- 减少过拟合
- 提高泛化能力

### 3.3 Dropout正则化

```python
# 卷积层后使用Dropout2d
nn.Dropout2d(0.3)

# 全连接层后使用Dropout
nn.Dropout(0.5)
```

**作用**：
- 随机丢弃神经元，防止过拟合
- 相当于训练多个子网络的集成

## 4. 损失函数优化

### 4.1 标签平滑 (Label Smoothing)

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**作用**：
- 防止模型对训练标签过于自信
- 提高泛化能力
- 减少过拟合

## 5. 优化器选择

### 5.1 AdamW优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2正则化
)
```

**作用**：
- AdamW比Adam有更好的权重衰减实现
- weight_decay提供L2正则化

## 6. 学习率调度 (Learning Rate Scheduling)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控val_loss
    factor=0.5,      # 降低为原来的50%
    patience=3,      # 3个epoch没改善就降低LR
    verbose=True
)

# 每个epoch后调用
scheduler.step(val_loss)
```

**作用**：
- 当训练停滞时自动降低学习率
- 帮助模型跳出局部最优
- 更精细地调整模型参数

## 7. 早停法 (Early Stopping)

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

**作用**：
- 防止过度训练
- 自动在最佳时机停止
- 节省训练时间

## 8. 梯度裁剪 (Gradient Clipping)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**作用**：
- 防止梯度爆炸
- 稳定训练过程

## 9. 迁移学习 (Transfer Learning)

使用预训练模型作为特征提取器：

```python
# 加载预训练VGG16
vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1')

# 冻结特征提取层
for param in vgg.features.parameters():
    param.requires_grad = False

# 替换分类器
vgg.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)
)
```

**作用**：
- 利用大规模数据集学到的特征
- 大幅提高准确率（从~20%提升到80%+）
- 减少训练时间和数据需求

## 10. 处理损坏图像

```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

**作用**：
- 处理数据集中的截断/损坏图像
- 避免训练中断

## 效果对比

| 指标 | 原始模型 | 改进后 | 提升 |
|-----|---------|-------|-----|
| 训练准确率 | 95% | 85% | -10% |
| 验证准确率 | 17% | 60%+ | +43% |
| 过拟合程度 | 严重 | 轻微 | 显著改善 |
| 参数量 | 26M | 5M | -81% |

## 最佳实践建议

1. **先尝试迁移学习**：如果有预训练模型可用，优先使用
2. **数据增强是关键**：在小数据集上尤其重要
3. **监控验证损失**：训练集准确率高不代表模型好
4. **使用早停**：避免过度训练
5. **合理设置学习率**：太大会不稳定，太小收敛太慢
6. **使用标准化**：ImageNet预训练模型必须使用ImageNet标准化参数

## 常见问题排查

| 问题 | 症状 | 解决方案 |
|-----|-----|---------|
| 过拟合 | 训练准确率高，验证准确率低 | 增加数据增强、Dropout、早停 |
| 欠拟合 | 训练和验证准确率都低 | 增加模型容量、训练更久 |
| 训练不稳定 | 损失波动大 | 降低学习率、梯度裁剪 |
| 收敛太慢 | 很多epoch后才有改善 | 提高学习率、检查数据标准化 |
