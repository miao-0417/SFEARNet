import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
def caclulate_weight(targets, num_classes):

    targets=targets.view(-1).detach().cpu().numpy()

    lis=Counter(targets)

    class_counts=np.zeros(num_classes)
    #print(class_counts)
    for i in range(num_classes):
        class_counts[i]=lis[i]
    # print(len(class_counts))
    #class_counts_max = max(class_counts).type(torch.float).cpu()
    class_counts_max = max(class_counts)
    class_weights = np.divide(class_counts_max, class_counts, out=np.zeros_like(class_counts),
                              where=class_counts != 0)
    weight=torch.tensor(class_weights).type(torch.float)
    #print(weight)
    return weight
