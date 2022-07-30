import os
import pytorch_lightning as pl

from .utils import instantiate_from_config, get_obj_from_str


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = instantiate_from_config(config['model'])
        
        self.criterions = {x['name']: instantiate_from_config(x) for x in config['criterions']}
        self.crit_weights = {x['name']: x['weight'] for x in config['criterions']}
        
        self.metrics = {x['name']: instantiate_from_config(x) for x in config['metrics']}

    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx, stage):       
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx):
        item = self._common_step(batch, batch_idx, 'train')
        return item

    def validation_step(self, batch, batch_idx):
        item = self._common_step(batch, batch_idx, 'valid')
        return item
    
    def test_step(self, batch, batch_idx):
        item = self._common_step(batch, batch_idx, 'test')
        return item

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        
        for item in self.config['optimizers']:
            optimizer = get_obj_from_str(item['target'])(
                self.parameters(), 
                **item.get('params', {}))
            optimizers.append(optimizer)
            
            if item.get('scheduler', None):
                scheduler = get_obj_from_str(item['scheduler']['target'])(
                    optimizer = optimizer, 
                    **item['scheduler'].get('params', {}))
                schedulers.append({
                    'scheduler':scheduler,
                    **item['scheduler']['additional'],
                })
        return optimizers, schedulers
    
    def configure_callbacks(self):
        callbacks = []
        
        for item in self.config.get('callbacks', []):
            params = item.get('params', {})
            if 'ModelCheckpoint' in item['target']:
                item['params'] = {
                    **params,
                    'dirpath': self.config['common']['save_path'],
                }

            callback = instantiate_from_config(item)
            callbacks.append(callback)  
        return callbacks

class MulticlassModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.use_bg = {x['name']: x.get('use_bg', True) for x in config['metrics']}
        
    def _common_step(self, batch, batch_idx, stage):
        gt_img, sg_mask, oh_mask = batch['image'], batch['sg_mask'].long(), batch['oh_mask'].float()
        pr_mask = self.model(gt_img.contiguous())
        # pr_mask = F.interpolate(pr_mask, size=batch['image'].shape[2:], mode='bilinear', align_corners=False)

        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](pr_mask, sg_mask) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_step=False, on_epoch=True, prog_bar=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            index = 0 if self.use_bg[m_name] else 1
            metric_value = self.metrics[m_name](pr_mask[:, index:, :, :], oh_mask[:, index:, :, :])
            self.log(metric_info, metric_value, on_step=False, on_epoch=True, prog_bar=True)              
        return {
            'loss': loss,
        }

class BinaryModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    
    def _common_step(self, batch, batch_idx, stage):
        gt_img, gt_mask = batch['image'], batch['mask'].float()
        pr_mask = self.model(gt_img.contiguous())
        # pr_mask = F.interpolate(pr_mask, size=batch['image'].shape[2:], mode='bilinear', align_corners=False)
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](pr_mask, gt_mask) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            metric_value = self.metrics[m_name](pr_mask, gt_mask)
            self.log(metric_info, metric_value, on_step=False, on_epoch=True, prog_bar=True)              
        return {
            'loss': loss,
        }