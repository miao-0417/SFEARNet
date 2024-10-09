import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss, self).__init__()
    def forward(self,inputs, targets,num_classes=3):
        # 将预测结果和真实标签转换为概率分布
        pred = F.softmax(inputs, dim=1)
        target = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        for class_idx in range(1,num_classes):
            # 提取当前类别的预测结果和真实标签
            pred_class = pred[:, class_idx, :, :]
            target_class = target[:, class_idx, :, :]

            # 计算Dice系数
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class)
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)

            # 将Dice系数转换为Dice Loss，并累加到总损失中
            class_loss = 1 - dice
            loss += class_loss

        # 对所有类别的Dice Loss进行平均
        loss /= (num_classes-1)

        return loss
