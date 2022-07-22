
import pytorch_lightning as pl

from utils import instantiate_from_config, get_obj_from_str


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = instantiate_from_config(config['model'])
        
        self.criterions = {x['name']: instantiate_from_config(x) for x in config['criterions']}
        self.crit_weights = {x['name']: x['weight'] for x in config['criterions']}
        
        self.metrics = {x['name']: instantiate_from_config(x) for x in config['metrics']}
        self.use_bg = {x['name']: x['use_bg'] for x in config['metrics']}

    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx, stage):
        gt_img, sg_mask, oh_mask = batch['image'], batch['sg_mask'].long(), batch['oh_mask']
        pr_msk = self.model(gt_img)
         
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](pr_msk, sg_mask) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_epoch=True, prog_bar=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            index = 0 if self.use_bg[m_name] else 1
            metric_value = self.metrics[m_name](pr_msk[:, index:, :, :], oh_mask[:, index:, :, :])
            self.log(metric_info, metric_value, on_epoch=True, prog_bar=True)              
        return {
            'loss': loss,
        }
    
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
            callback = instantiate_from_config(item)
            callbacks.append(callback)  
        return callbacks