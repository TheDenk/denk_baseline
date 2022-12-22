import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalWithLogitsFirst(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalWithLogitsFirst, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-12

    def forward(self, probs, target):
        probs = torch.sigmoid(probs)
        one_subtract_probs = 1.0 - probs
        # add epsilon
        probs_new = probs + self.epsilon
        one_subtract_probs_new = one_subtract_probs + self.epsilon
        # calculate focal loss
        log_pt = target * torch.log(probs_new) + (1.0 - target) * torch.log(one_subtract_probs_new)
        pt = torch.exp(log_pt)
        focal_loss = -1.0 * (self.alpha * (1 - pt) ** self.gamma) * log_pt
        return torch.mean(focal_loss)
    

class FocalWithLogitsSecond(nn.Module):
    def __init__(self, alpha=.25, gamma=2, device='cpu'):
        super(FocalWithLogitsSecond, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
