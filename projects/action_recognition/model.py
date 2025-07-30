from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.vjepa2 import VJEPA2Config, VJEPA2PreTrainedModel, VJEPA2Model
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler, ImageClassifierOutput


configs = {
    "vit_large": VJEPA2Config(
        crop_size=256,
        frames_per_clip=64,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        mlp_ratio=4,
        pred_hidden_size=384,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=12,
        pred_num_mask_tokens=10,
    ),
    "vit_huge": VJEPA2Config(
        crop_size=256,
        frames_per_clip=64,
        hidden_size=1280,
        num_attention_heads=16,
        num_hidden_layers=32,
        mlp_ratio=4,
        pred_hidden_size=384,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=12,
        pred_num_mask_tokens=10,
    ),
    "vit_giant": VJEPA2Config(
        crop_size=256,
        frames_per_clip=64,
        hidden_size=1408,
        num_attention_heads=22,
        num_hidden_layers=40,
        mlp_ratio=48 / 11,
        pred_hidden_size=384,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=12,
        pred_num_mask_tokens=10,
    ),
    "vit_giant_384": VJEPA2Config(
        crop_size=384,
        frames_per_clip=64,
        hidden_size=1408,
        num_attention_heads=22,
        num_hidden_layers=40,
        mlp_ratio=48 / 11,
        pred_hidden_size=384,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=12,
        pred_num_mask_tokens=10,
    )
}


class VJEPA2ForVideoClassification(VJEPA2PreTrainedModel):
    def __init__(self, config_name: str, num_classes: int, pretrained_name_or_path: str = None):
        super().__init__(configs[config_name])
        
        if pretrained_name_or_path:
            self.vjepa2 = VJEPA2Model.from_pretrained(
                pretrained_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )
        else:
            self.vjepa2 = VJEPA2Model(configs[config_name])
        self.pooler = VJEPA2AttentivePooler(configs[config_name])
        # self.vjepa2.requires_grad_(False)
        # self.pooler.requires_grad_(False)
        
        self.classifier = nn.Sequential(
            # nn.Linear(configs[config_name].hidden_size, configs[config_name].hidden_size, bias=True),
            # nn.SiLU(),
            nn.Linear(configs[config_name].hidden_size, num_classes, bias=True),
        )
        self.post_init()

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        outputs = self.vjepa2(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = outputs.last_hidden_state
        pooler_output = self.pooler(last_hidden_state)
        logits = self.classifier(pooler_output)
        return logits


class FocalWithLogits(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalWithLogits, self).__init__()
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


class MultiheadVJEPA2ForVideoClassification(VJEPA2PreTrainedModel):
    def __init__(self, config_name: str, num_classes: int, num_heads: int, pretrained_name_or_path: str = None):
        super().__init__(configs[config_name])
        
        if pretrained_name_or_path:
            self.vjepa2 = VJEPA2Model.from_pretrained(
                pretrained_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )
        else:
            self.vjepa2 = VJEPA2Model(configs[config_name])
        self.pooler = VJEPA2AttentivePooler(configs[config_name])
        
        self.classifier = nn.ModuleList([
            nn.Linear(configs[config_name].hidden_size, num_classes, bias=True)
        for _ in range(num_heads)])

        self.loss_fn = FocalWithLogits()
        self.post_init()

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: torch.Tensor = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        outputs = self.vjepa2(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = outputs.last_hidden_state
        pooler_output = self.pooler(last_hidden_state)
        head_logits = torch.stack([head(pooler_output) for head in self.classifier], dim=1)
        batch, num_heads, num_classes = head_logits.shape
        logits = head_logits.mean(1)
        
        if labels is not None:
            head_logits = head_logits.reshape(batch * num_heads, num_classes)
            labels_logits = labels.reshape(batch * num_heads, num_classes)
            loss = self.loss_fn(head_logits, labels_logits)
            return loss.mean(-1), logits
        
        return logits