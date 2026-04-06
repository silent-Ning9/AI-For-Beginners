# 图像分割学习体会

## 一、分割任务概述

图像分割是像素级别的分类任务，为图像中的**每个像素**预测其所属类别。

### 分割类型对比

| 类型 | 描述 | 示例 |
|------|------|------|
| **语义分割** | 对每个像素分类，同类像素不区分实例 | 将图像分为"人"、"车"、"背景" |
| **实例分割** | 区分同类对象的不同实例 | 识别出图像中的每一只羊 |

### 与目标检测的区别

| 任务 | 输出 | 精度 |
|------|------|------|
| 目标检测 | 边界框 + 类别 | 粗粒度定位 |
| 图像分割 | 像素级掩码 | 精确到像素 |

## 二、网络架构

分割网络采用**编码器-解码器**结构：

```
输入图像 → 编码器(特征提取) → 瓶颈层 → 解码器(上采样) → 分割掩码
```

### 核心组件

| 组件 | 功能 |
|------|------|
| **编码器** | 卷积 + 池化，逐步下采样提取特征 |
| **瓶颈层** | 最深层特征表示 |
| **解码器** | 反卷积/上采样，恢复空间分辨率 |

## 三、主流模型

### 1. SegNet

最基础的编解码器架构：
- 编码器：使用 VGG 风格的卷积和池化
- 解码器：使用池化索引进行上采样
- 特点：结构简单，但精度有限

### 2. U-Net

医学影像分割的经典架构：

```
编码器                    解码器
  ↓                         ↑
[Conv] → → → → → → → → → [Concat]  ← 跳跃连接
  ↓                         ↑
[Pool]                   [Upsample]
  ↓                         ↑
[Conv] → → → → → → → → [Concat]
  ↓                         ↑
  ...                      ...
```

**核心创新：跳跃连接（Skip Connections）**
- 将编码器特征直接连接到解码器
- 保留空间位置信息
- 解决下采样导致的细节丢失

### 3. DeepLab 系列

- 使用**空洞卷积（Atrous Convolution）**扩大感受野
- 引入 **ASPP（Atrous Spatial Pyramid Pooling）**
- 适合多尺度目标分割

## 四、损失函数

### 二元分割

**Binary Cross-Entropy Loss (BCE)**
```python
loss = nn.BCEWithLogitsLoss()  # 包含 Sigmoid
```

### 多类分割

**Cross-Entropy Loss**
```python
loss = nn.CrossEntropyLoss()
```

### 类别不平衡处理

- **加权 BCE**：给少数类更高权重
- **Dice Loss**：直接优化 Dice 系数
- **Focal Loss**：降低简单样本的权重

## 五、评估指标

### 1. 像素准确率 (Pixel Accuracy)

```
Pixel Acc = 正确分类的像素数 / 总像素数
```

### 2. 交并比 (IoU / Jaccard Index)

```
IoU = 预测与真实交集面积 / 预测与真实并集面积
```

### 3. Dice 系数 (F1 Score)

```
Dice = 2 × |预测 ∩ 真实| / (|预测| + |真实|)
```

### 指标对比

| 指标 | 范围 | 特点 |
|------|------|------|
| Pixel Acc | 0-1 | 对类别不平衡敏感 |
| IoU | 0-1 | 对形状敏感 |
| Dice | 0-1 | 对重叠区域敏感 |

## 六、实验总结

### 本次实验：人体分割

使用 U-Net 在 Segmentation Full Body MADS Dataset 上进行人体轮廓分割。

#### 实现要点

```python
# U-Net 跳跃连接
d0 = self.up0(b)           # 上采样
d0 = torch.cat([d0, e3], dim=1)  # 拼接编码器特征
d0 = self.dec_conv0(d0)    # 卷积处理
```

```python
# Dice 系数计算
def dice_coefficient(pred, target):
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum())
```

#### 训练技巧

1. **数据预处理**
   - 统一图像尺寸 (256×256)
   - 掩码二值化 (threshold=0.5)

2. **优化策略**
   - Adam 优化器，学习率 1e-3
   - ReduceLROnPlateau 学习率调度
   - Batch Normalization 加速训练

3. **评估方法**
   - Dice > 0.9 表示优秀分割
   - IoU > 0.8 表示良好的边界检测

## 七、常见问题与解决方案

### 1. 边界模糊

**原因**：下采样丢失细节
**解决**：使用跳跃连接、增加边缘损失

### 2. 类别不平衡

**原因**：前景/背景比例悬殊
**解决**：加权损失、Dice Loss、过采样

### 3. 小目标分割困难

**原因**：特征图分辨率不足
**解决**：减少下采样层、使用空洞卷积

### 4. 显存不足

**原因**：分割任务计算量大
**解决**：
- 减小 batch_size
- 使用混合精度训练
- 滑动窗口处理大图

## 八、应用领域

| 领域 | 应用 |
|------|------|
| **医学影像** | 肿瘤分割、器官检测、细胞计数 |
| **自动驾驶** | 车道检测、可行驶区域分割 |
| **视频制作** | 虚拟背景、视频抠像 |
| **遥感图像** | 土地利用分类、建筑物检测 |
| **工业检测** | 缺陷检测、产品分割 |

## 九、进阶方向

1. **实例分割**
   - Mask R-CNN
   - YOLACT
   - SOLOv2

2. **全景分割**
   - 统一语义和实例分割
   - Panoptic FPN

3. **弱监督分割**
   - 使用边界框标注
   - 使用图像级标签

4. **实时分割**
   - BiSeNet
   - Fast-SCNN
   - STDC

## 十、参考资料

- [U-Net 论文](https://arxiv.org/pdf/1505.04597.pdf)
- [DeepLab v3+ 论文](https://arxiv.org/pdf/1802.02611.pdf)
- [PyTorch Segmentation 教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)

---

*学习日期：2026年4月*
