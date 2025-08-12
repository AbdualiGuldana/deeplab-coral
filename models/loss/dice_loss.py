import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.7, ce_weight=0.3, smooth=1.):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long())

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # batch, H, W
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        cardinality = torch.sum(inputs_soft + targets_one_hot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_per_class.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# class DiceFocalLoss(nn.Module):
#     def __init__(self):
#         super(DiceFocalLoss, self).__init__()

#     def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1.):
#         # Cross-entropy loss calculation
#         # F.cross_entropy expects inputs (logits) of shape (B, C, H, W)
#         # and targets (class indices) of shape (B, H, W).
#         # Your code is already set up to receive this.
#         ce_loss = F.cross_entropy(inputs, targets.long(), reduction='mean')
        
#         # Dice loss calculation
#         # The Dice loss requires one-hot encoded targets and probabilities.
        
#         # 1. Get probabilities from logits
#         inputs_prob = F.softmax(inputs, dim=1)
        
#         # 2. One-hot encode targets to match the input shape
#         num_classes = inputs_prob.shape[1]
#         targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
#         # 3. Flatten for Dice loss
#         inputs_prob_flat = inputs_prob.contiguous().view(-1)
#         targets_one_hot_flat = targets_one_hot.contiguous().view(-1)
        
#         # 4. Calculate Dice loss
#         intersection = (inputs_prob_flat * targets_one_hot_flat).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs_prob_flat.sum() + targets_one_hot_flat.sum() + smooth)
        
#         # 5. Combine the losses with a weighted sum
#         # Adjust the weights (e.g., 0.5) to balance the contribution of each loss.
#         total_loss = 0.5 * ce_loss + 0.5 * dice_loss
        
#         return total_loss
    
# import torch
# from torch import Tensor
# from torch import nn
# from torch.nn import functional as F
# from torchvision.ops.focal_loss import sigmoid_focal_loss


# class DiceFocalLoss(nn.Module):
#     def __init__(self):
#         super(DiceFocalLoss, self).__init__()
    
#     def forward(self, inputs: Tensor, targets: Tensor, smooth: float=1.):
#         inputs = F.sigmoid(inputs)
#         # inputs = torch.argmax(inputs, dim=1)
#         inputs = inputs.view(-1) # .to(torch.float32)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
#         cls_loss = F.cross_entropy(inputs, targets, reduction='mean')
#         # cls_loss = sigmoid_focal_loss(inputs, targets, reduction='mean')

#         losses = dict(
#             cls_loss=cls_loss,
#             # dice_loss=dice_loss
#         )
#         # loss = ce_loss + dice_loss

#         return losses