import torch.nn as nn
class CELoss_Weighted(nn.Module):
    def __init__(self, weight=None):
        super(CELoss_Weighted, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        loss = nn.CrossEntropyLoss(weight=self.weight)
        return loss(inputs, targets)