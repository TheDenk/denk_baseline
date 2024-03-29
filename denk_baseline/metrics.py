import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


class BaseMetric:
    def __init__(self, from_logits=True, threshold=0.5, **kwargs):
        self.from_logits = from_logits
        self.threshold = threshold
        self.kwargs = kwargs

    def calculate(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, y_pred, y_true):
        # if self.from_logits:
        #     y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        m_value = self.calculate(y_true, y_pred, **self.kwargs)
        return m_value or 0.0


class ROCAUC(BaseMetric):
    def calculate(self, y_true, y_pred, **kwargs):
        return roc_auc_score(y_true, y_pred, **self.kwargs)


class Precision(BaseMetric):
    def calculate(self, y_true, y_pred, **kwargs):
        return precision_score(y_true, y_pred, **self.kwargs)


class Recall(BaseMetric):
    def calculate(self, y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, **self.kwargs)


class F1Score(BaseMetric):
    def calculate(self, y_true, y_pred, **kwargs):
        return f1_score(y_true, y_pred, **self.kwargs)


class MeanAccuracyScore(BaseMetric):
    def calculate(self, y_true, y_pred, **kwargs):
        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=list(range(1024)))

        cls_cnt = conf_matrix.sum(axis=1) 
        cls_hit = np.diag(conf_matrix)

        metrics = [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
        mean_class_acc = np.mean(metrics)
        return mean_class_acc


class PFScore:
    def __init__(self, beta=1, from_logits=True, threshold=None):
        self.beta = beta
        self.from_logits = from_logits
        self.threshold = threshold

    def __call__(self, y_pred, y_true):
        return self.calculate(y_pred, y_true, self.beta)

    def calculate(self, preds, labels, beta=1):
        if self.from_logits:
            preds = torch.sigmoid(preds)
        preds = preds.clip(0, 1)
        if self.threshold is not None:
            preds = (preds > self.threshold).long()
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
