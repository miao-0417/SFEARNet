import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
    def forward(self, inputs, targets,num_classes=3):
        pred = F.softmax(inputs, dim=1)
        target = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        for class_idx in range(1,num_classes):
            # 提取当前类别的预测结果和真实标签
            pred_class = pred[:, class_idx, :, :]
            target_class = target[:, class_idx, :, :]

            # 计算交集和并集
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class) - intersection

            # 计算IOU值
            iou = intersection / (union + 1e-7)  # 添加一个小的常数以避免除以零

            # 将IOU值转换为IOU Loss，并累加到总损失中
            class_loss = 1 - iou
            loss += class_loss

        # 对所有类别的IOU Loss进行平均
        loss /= (num_classes-1)

        return loss