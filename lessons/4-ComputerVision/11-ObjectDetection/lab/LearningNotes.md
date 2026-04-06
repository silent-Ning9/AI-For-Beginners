# 目标检测学习体会

## 一、目标检测概述

目标检测是计算机视觉中的核心任务，与图像分类不同，它不仅要识别图像中有哪些物体，还要定位它们的位置（边界框坐标）。

### 与图像分类的区别

| 任务 | 输入 | 输出 |
|------|------|------|
| 图像分类 | 图像 | 类别标签 |
| 目标检测 | 图像 | 类别标签 + 边界框坐标 |

## 二、核心概念

### 1. 边界框 (Bounding Box)

边界框用四个坐标表示：`(xmin, ymin, xmax, ymax)`，分别代表左上角和右下角的坐标。

### 2. 交并比 (IoU - Intersection over Union)

IoU 用于衡量预测框与真实框的重叠程度：

```
IoU = 交集面积 / 并集面积
```

- IoU = 1：完全重叠
- IoU = 0：完全不重叠
- 通常 IoU > 0.5 被认为是有效的检测结果

### 3. 平均精度 (mAP)

mAP 是目标检测的标准评估指标：
- **Precision（精度）**：预测为正例中真正的正例比例
- **Recall（召回率）**：真正正例中被预测为正例的比例
- **AP**：Precision-Recall 曲线下的面积
- **mAP**：所有类别的 AP 平均值

## 三、主流算法

### 1. 两阶段检测器

#### R-CNN 系列
- **R-CNN**：选择性搜索生成候选区域 → CNN 特征提取 → SVM 分类
- **Fast R-CNN**：共享卷积特征，一次前向传播
- **Faster R-CNN**：引入区域建议网络 (RPN)，端到端训练

优点：精度高
缺点：速度相对较慢

### 2. 单阶段检测器

#### YOLO (You Only Look Once)
- 将图像分成 S×S 网格
- 每个网格预测边界框和类别
- 实时检测，速度极快

#### SSD (Single Shot Detector)
- 多尺度特征图检测
- 平衡速度与精度

#### RetinaNet
- 引入 Focal Loss 解决正负样本不平衡
- 单阶段检测器中的精度标杆

## 四、实践总结

### 本次实验：行人检测

使用 Faster R-CNN + ResNet-50 FPN 骨干网络在 PennFudanPed 数据集上进行行人检测。

#### 关键步骤

1. **数据准备**
   - 从分割掩码提取边界框
   - 自定义 Dataset 类处理标注

2. **模型选择**
   - 使用预训练的 Faster R-CNN
   - 替换分类头适应新类别数

3. **训练技巧**
   - 迁移学习：利用预训练权重
   - 学习率调度：StepLR 逐步降低学习率
   - 数据增强：可进一步提升性能

4. **评估指标**
   - IoU 阈值：通常设为 0.5
   - 置信度阈值：过滤低置信度预测

### 代码要点

```python
# 自定义数据集关键代码
class PennFudanDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # 从掩码提取边界框
        pos = np.where(masks[i])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
```

```python
# 模型修改
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

## 五、踩坑与解决方案

### 1. 数据格式问题
- **问题**：不同数据集标注格式不同（COCO、VOC、自定义）
- **解决**：统一转换为 PyTorch 需要的字典格式

### 2. 显存不足
- **问题**：大模型或大图像导致 OOM
- **解决**：减小 batch_size，或使用梯度累积

### 3. 训练不稳定
- **问题**：Loss 震荡或发散
- **解决**：降低学习率，使用学习率预热

### 4. 预测框过多
- **问题**：NMS 后仍有大量重叠框
- **解决**：调整置信度阈值和 NMS 的 IoU 阈值

## 六、进一步学习方向

1. **尝试其他架构**
   - YOLOv5/v8：实时检测
   - DETR：基于 Transformer 的检测器

2. **数据增强**
   - 随机裁剪、翻转、颜色抖动
   - Mosaic、MixUp 等高级增强

3. **模型优化**
   - 量化加速推理
   - 模型剪枝减小体积

4. **部署应用**
   - ONNX 导出跨平台部署
   - TensorRT 加速

## 七、参考资料

- [Faster R-CNN 论文](https://arxiv.org/pdf/1506.01497.pdf)
- [PyTorch 官方目标检测教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [YOLO 官方网站](https://pjreddie.com/darknet/yolo/)
- [COCO 数据集](https://cocodataset.org/)

---

*学习日期：2026年4月*
