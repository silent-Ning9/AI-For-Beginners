# 迁移学习 (Transfer Learning) 指南

## 1. 什么是迁移学习？

迁移学习是一种机器学习技术，将从一个任务（源域）学到的知识应用到另一个相关任务（目标域）中。

```
传统学习：每个任务从零开始学习
任务A → 训练模型A（从随机权重开始）
任务B → 训练模型B（从随机权重开始）

迁移学习：复用已学到的知识
ImageNet分类 → 预训练模型 → 微调 → Pet分类
```

## 2. 为什么迁移学习有效？

### 2.1 特征的层次性

卷积神经网络学习到的特征具有层次性：

| 层级 | 学习的特征 | 可视化示例 | 通用性 |
|-----|-----------|-----------|-------|
| 第1层 | 边缘、线条 | ─ │ ╱ ╲ | 极高 |
| 第2层 | 纹理、简单形状 | ○ □ △ | 很高 |
| 第3层 | 局部模式 | 眼睛、轮子 | 较高 |
| 第4层 | 物体部件 | 狗头、猫耳 | 中等 |
| 第5层 | 完整物体 | 狗、猫、车 | 较低 |

**关键洞察**：底层特征（边缘、纹理）在所有视觉任务中都有用，无需重新学习。

### 2.2 知识迁移原理

```
源任务（ImageNet，1000类，120万张图片）
    ↓
学到通用视觉特征：边缘检测、纹理识别、形状识别
    ↓
迁移到目标任务（Pet分类，37类，7000张图片）
    ↓
只需学习：如何将这些特征映射到新的类别
```

## 3. 迁移学习的三种策略

### 3.1 特征提取（Feature Extraction）

**冻结所有卷积层，只训练分类器**

```python
# 加载预训练模型
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

# 只优化分类器参数
optimizer = Adam(vgg.classifier.parameters(), lr=0.001)
```

**适用场景**：
- 目标数据集较小
- 与源任务相似
- 计算资源有限

### 3.2 微调（Fine-tuning）

**解冻部分或全部层，使用较小学习率训练**

```python
# 加载预训练模型
vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1')

# 替换分类器
vgg.classifier[6] = nn.Linear(4096, num_classes)

# 解冻最后几层卷积层
for param in vgg.features[-4:].parameters():
    param.requires_grad = True

# 使用较小学习率
optimizer = Adam(vgg.parameters(), lr=0.0001)
```

**适用场景**：
- 目标数据集较大
- 与源任务有一定差异
- 需要更高精度

### 3.3 渐进式解冻

**逐步解冻层，从顶层到底层**

```python
# 第1阶段：只训练分类器（冻结所有卷积层）
for param in vgg.features.parameters():
    param.requires_grad = False
train(epochs=5)

# 第2阶段：解冻最后2个卷积块
for param in vgg.features[-2:].parameters():
    param.requires_grad = True
train(epochs=5)

# 第3阶段：解冻所有层
for param in vgg.features.parameters():
    param.requires_grad = True
train(epochs=10)
```

## 4. 迁移学习 vs 从头训练

### 4.1 对比表

| 方面 | 从头训练 | 迁移学习 |
|-----|---------|---------|
| **初始化** | 随机权重 | 预训练权重 |
| **数据需求** | 大量（>10,000张） | 少量（几百张即可） |
| **训练时间** | 长（数小时到数天） | 短（几分钟到几小时） |
| **收敛速度** | 慢 | 快 |
| **最终精度** | 取决于数据和模型 | 通常更高 |
| **过拟合风险** | 高（小数据集） | 低 |
| **计算成本** | 高 | 低 |

### 4.2 训练曲线对比

```
验证准确率
    │
90% │          ──── 迁移学习（快速收敛到高精度）
    │        ╱╱╱╱
    │       ╱
    │      ╱
60% │    ╱
    │   ╱        ╱╱╱╱╱╱ 从头训练（缓慢收敛）
    │  ╱       ╱
30% │ ╱      ╱
    │╱    ╱
    └─────────────────────────────→ Epoch
       5    10    15    20    30
```

## 5. 常用预训练模型

| 模型 | 参数量 | 特点 | 适用场景 |
|-----|-------|-----|---------|
| VGG16 | 138M | 结构简单，效果好 | 通用图像分类 |
| ResNet50 | 25.6M | 残差连接，训练稳定 | 通用图像分类 |
| MobileNetV2 | 3.5M | 轻量级，速度快 | 移动端部署 |
| EfficientNet-B0 | 5.3M | 效率高，精度好 | 资源受限场景 |
| ViT | 86M-632M | Transformer架构 | 大规模数据集 |

### 加载预训练模型

```python
import torchvision.models as models

# VGG16
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# ResNet50
resnet50 = models.resnet50(weights='IMAGENET1K_V1')

# MobileNetV2
mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')

# EfficientNet
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
```

## 6. 关键技术细节

### 6.1 数据标准化

**必须使用ImageNet的均值和标准差**：

```python
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet均值
    std=[0.229, 0.224, 0.225]    # ImageNet标准差
)
```

### 6.2 学习率设置

```python
# 特征提取：可以使用较高学习率
optimizer = Adam(classifier.parameters(), lr=0.001)

# 微调：使用较低学习率
optimizer = Adam(model.parameters(), lr=0.0001)

# 差异化学习率
optimizer = Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},  # 卷积层
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # 分类器
])
```

### 6.3 图像尺寸

```python
# 标准输入尺寸
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # 224x224是标准尺寸
    transforms.ToTensor(),
])
```

## 7. 实战案例

### 7.1 完整的迁移学习流程

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 1. 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')

# 2. 冻结特征提取层
for param in model.parameters():
    param.requires_grad = False

# 3. 替换分类器
num_classes = 37  # Pet数据集类别数
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# 4. 只优化分类器参数
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 5. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 6. 训练
for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 8. 效果对比（Oxford Pets数据集）

| 方法 | 训练参数 | 训练时间 | 验证准确率 |
|-----|---------|---------|-----------|
| 从头训练（简单CNN） | 26M | 50 epochs | 30-50% |
| VGG16特征提取 | 25M | 5 epochs | 80-85% |
| ResNet50特征提取 | 2M | 5 epochs | 82-87% |
| ResNet50微调 | 25M | 10 epochs | 88-92% |

## 9. 何时使用迁移学习？

### 推荐使用迁移学习的情况

- ✅ 目标数据集较小（<10,000张图片）
- ✅ 任务与ImageNet相似（都是自然图像）
- ✅ 计算资源有限
- ✅ 需要快速得到结果

### 可能需要从头训练的情况

- ⚠️ 数据集非常大（>100,000张图片）
- ⚠️ 任务与ImageNet差异大（医学图像、卫星图像）
- ⚠️ 需要完全定制的架构
- ⚠️ 有充足的计算资源和时间

## 10. 常见问题

### Q1: 为什么冻结层可以加速训练？

**A**: 冻结层不需要计算梯度，减少了反向传播的计算量。同时，预训练的特征已经很好，只需训练分类器即可。

### Q2: 应该冻结多少层？

**A**: 一般原则：
- 数据少：冻结所有卷积层
- 数据中等：解冻最后1-2个卷积块
- 数据多：可以全部微调

### Q3: 为什么验证准确率比训练准确率高？

**A**: 这是迁移学习的正常现象：
1. 训练时有Dropout正则化
2. BatchNorm在训练和验证时行为不同
3. 预训练特征本身质量很高

### Q4: 不同预训练模型如何选择？

**A**: 
- 追求精度：ResNet101, EfficientNet-B7
- 追求速度：MobileNetV2, EfficientNet-B0
- 平衡选择：ResNet50, VGG16

## 11. 总结

| 核心要点 | 说明 |
|---------|-----|
| **核心思想** | 复用已学到的通用特征 |
| **关键技术** | 冻结层 + 替换分类器 |
| **主要优势** | 少数据、快训练、高精度 |
| **关键参数** | 使用ImageNet标准化 |
| **最佳实践** | 优先尝试特征提取，效果不好再微调 |

---

**参考资料**：
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n: Transfer Learning](http://cs231n.stanford.edu/)
