import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import instantiate_from_config, get_obj_from_str


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


def calc_weights(cls_num_list, beta=0.9999, device='cpu'):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    return per_cls_weights


class FocalWithLogitsThird(nn.Module):
    def __init__(self, cls_num_list=None, gamma=0.):
        super(FocalWithLogitsThird, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = cls_num_list if cls_num_list is None else calc_weights(cls_num_list, device='cuda:0')

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, device='cuda:0'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.device = device
        self.weight = cls_num_list if cls_num_list is None else calc_weights(cls_num_list, device=self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.to(dtype=torch.bfloat16, device=self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    

class DistillationLossTwoHeads(torch.nn.Module):
    """
    https://github.com/Amshaker/SwiftFormer/blob/main/util/losses.py
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, config):
        super().__init__()
        self.base_criterion = instantiate_from_config(config['criterion'])
        self.teacher_model = instantiate_from_config(config['teacher_model']).eval().to(dtype=torch.bfloat16, device='cuda')
        assert config['distillation_type'] in ['none', 'mse', 'soft', 'hard']
        self.distillation_type = config['distillation_type']
        self.alpha = config['alpha']
        self.tau = config['tau']

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # Don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))
        elif self.distillation_type == 'mse':
            distillation_loss = F.mse_loss(outputs_kd, teacher_outputs)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class DistillationLossOneHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_criterion = instantiate_from_config(config['criterion'])
        self.teacher_model = instantiate_from_config(config['teacher_model']).eval().to(dtype=torch.bfloat16, device='cuda')
        assert config['distillation_type'] in ['none', 'rmse', 'soft', 'hard']
        self.distillation_type = config['distillation_type']
        self.alpha = config['alpha']
        self.tau = config['tau']

    def forward(self, inputs, outputs, labels):
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))
        elif self.distillation_type == 'rmse':
            distillation_loss = torch.sqrt(F.mse_loss(outputs, teacher_outputs) + 1e-6)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss