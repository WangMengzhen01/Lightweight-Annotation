import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Binary Focal Loss
        Args:
            alpha: 控制正负样本权重的平衡因子 (默认0.25)
            gamma: 聚焦参数，控制难易样本的权重 (默认2.0)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, 1, H, W] -> 模型直接输出的 logits (未经过 Sigmoid)
        # targets: [Batch, 1, H, W] -> 标签 (0 或 1)

        # 1. 计算标准的 Binary Cross Entropy Loss (不进行归约，保留每个像素的loss)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2. 计算 pt (预测概率)
        # p_t = exp(-bce_loss) 这是一个数学技巧，对于二分类 logits 成立
        pt = torch.exp(-bce_loss)

        # 3. 计算 Focal Loss 公式: alpha * (1-pt)^gamma * BCE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss