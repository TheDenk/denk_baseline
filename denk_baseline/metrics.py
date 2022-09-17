import torch


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