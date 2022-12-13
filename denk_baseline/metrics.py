import torch
import torch.nn.functional as F


class DiceCoef:
    def __init__(self, threshold=0.5, dim=(2,3), eps=0.00001):
        self.threshold = threshold
        self.dim = dim
        self.eps = eps

    def __call__(self, y_pred, y_true):
        return self.calculate(y_pred, y_true)
    
    def calculate(self, y_pred, y_true):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > self.threshold).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=self.dim)
        union = y_true.sum(dim=self.dim) + y_pred.sum(dim=self.dim)
        dice = ((2 * inter + self.eps) / (union + self.eps)).mean(dim=(1,0))
        return dice


class PFScoreFromLogits:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, y_pred, y_true):
        return self.calculate(y_pred, y_true, self.beta)

    def calculate(self, preds, labels, beta=1):
        # preds = F.sigmoid(preds)
        preds = preds.clip(0, 1)
        y_true_count = labels.sum()
        ctp = preds[labels==1].sum()
        cfp = preds[labels==0].sum()
        beta_squared = beta * beta
        c_precision = ctp / (ctp + cfp)
        c_recall = ctp / y_true_count
        if (c_precision > 0 and c_recall > 0):
            result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
            return result
        else:
            return 0.0
