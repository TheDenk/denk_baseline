import torch
import pytorch_lightning as pl
from .utils import instantiate_from_config, get_obj_from_str


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = instantiate_from_config(config['model'])
        if self.config['model'].get('weights', False):
            self.model = self.load_checkpoint(self.config['model']['weights'])

        self.batch_augs = config.get('batch_augs', False)
        if self.batch_augs:
            self.batch_augs = instantiate_from_config(self.batch_augs)
        else:
            self.batch_augs = torch.nn.Identity()
        
        self.criterions = {x['name']: instantiate_from_config(x) for x in config['criterions']}
        self.crit_weights = {x['name']: x['weight'] for x in config['criterions']}
        
        self.metrics = {x['name']: instantiate_from_config(x) for x in config['metrics']}

        self.save_hyperparameters()

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
                    # **item['scheduler']['additional'],
                })
        return optimizers, schedulers
    
    def configure_callbacks(self):
        callbacks = []
        
        for item in self.config.get('callbacks', []):
            params = item.get('params', {})
            if 'ModelCheckpoint' in item['target']:
                item['params'] = {
                    **params,
                    'dirpath': self.config['general']['save_path'],
                }

            callback = instantiate_from_config(item)
            callbacks.append(callback)  
        return callbacks

    def load_checkpoint(self, config):
        ckpt = torch.load(config['checkpoint'], map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        if 'ignore_layers' in ckpt:
            state_dict = {}
            for name, params in ckpt.items():
                if name in config['ignore_layers']:
                    continue
                state_dict[name] = params

            print('LOAD MODEL: ', self.load_state_dict(state_dict, strict=False))
        else:
            print('LOAD MODEL: ', self.load_state_dict(ckpt, strict=False))


class SegmentationMulticlassModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.use_bg = {x['name']: x.get('use_bg', True) for x in config['metrics']}
        
    def _common_step(self, batch, batch_idx, stage):
        gt_img, sg_mask, oh_mask = batch['image'], batch['sg_mask'], batch['oh_mask']
        pr_mask = self.model(gt_img.contiguous())
        # print(gt_img.shape, pr_mask.shape, oh_mask.shape, sg_mask.shape)
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](pr_mask, sg_mask) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            index = 0 if self.use_bg[m_name] else 1
            metric_value = self.metrics[m_name](pr_mask[:, index:, :, :], oh_mask[:, index:, :, :])
            self.log(metric_info, metric_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {
            'loss': loss,
        }


class SegmentationBinaryModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        gt_img, gt_mask = batch['image'], batch['mask'].float()
        pr_mask = self.model(gt_img.contiguous()).float()
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](pr_mask, gt_mask) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            metric_value = self.metrics[m_name](pr_mask, gt_mask)
            self.log(metric_info, metric_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)              
        return {
            'loss': loss,
        }

class ClassificationBase(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.metric_values = {
            x: {'gt': [], 'pr': []} for x in ['train', 'valid', 'test']
        }

    def on_validation_epoch_start(self):
        self.metric_values['valid'] = {'gt': [], 'pr': []}
    
    def on_train_epoch_start(self):
        self.metric_values['train'] = {'gt': [], 'pr': []}

    def on_test_epoch_start(self):
        self.metric_values['test'] = {'gt': [], 'pr': []}    
    
    def calculate_metrics(self, stage):
        gt = self.metric_values[stage]['gt']
        pr = self.metric_values[stage]['pr']
        
        metrics = self.config.get('metrics', [])
        for m_info in metrics:
            m_name = m_info['name']
            metric = instantiate_from_config(m_info)
            metric_value = metric(torch.cat(pr), torch.cat(gt))
            self.log(f'{m_name}_{stage}', metric_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        metrics_thresholds = self.config.get('metrics_thresholds', False)
        if metrics_thresholds:
            monitor_name = metrics_thresholds['monitor']
            thresholds = metrics_thresholds['thresholds']
            best_metric = 0.0
            best_thres = 0.0
            
            mon_target = metrics_thresholds['metrics'][monitor_name]['target']
            mon_params = metrics_thresholds['metrics'][monitor_name].get('params', {})

            for thres in thresholds:
                mon_object = get_obj_from_str(mon_target)(**mon_params, threshold=thres)
                mon_value = mon_object(torch.cat(pr), torch.cat(gt))
                if metric_value > best_metric:
                    best_metric = mon_value
                    best_thres = thres

            for metric_name, metric_info in metrics_thresholds['metrics'].items():
                metric = get_obj_from_str(metric_info['target'])(**metric_info.get('params', {}), threshold=best_thres)
                metric_value = metric(torch.cat(pr), torch.cat(gt))
                self.log(f'{metric_name}_{stage}', metric_value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f'{stage}_best_threshold', best_thres, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.calculate_metrics('valid')
    
    def on_train_epoch_end(self):
        self.calculate_metrics('train')

    def on_test_epoch_end(self):
        self.calculate_metrics('test')


class ClassificationBinaryModel(ClassificationBase):
    def _common_step(self, batch, batch_idx, stage):
        with torch.autograd.set_detect_anomaly(True):
            gt_img, gt_label = batch['image'], batch['label'].float().unsqueeze(1)
            if self.training:
                gt_img = self.batch_augs(gt_img)
            pr_label = self.model(gt_img.contiguous()).float()

            loss = 0
            for c_name in self.criterions.keys():
                c_loss = self.criterions[c_name](pr_label, gt_label) * self.crit_weights[c_name]
                self.log(f'{c_name}_loss_{stage}', c_loss, on_epoch=True, prog_bar=True, sync_dist=True)
                loss += c_loss
            self.log(f'total_loss_{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
            self.metric_values[stage]['pr'].append(pr_label.cpu().detach().squeeze())
            self.metric_values[stage]['gt'].append(gt_label.cpu().detach().squeeze().long())

        return {
            'loss': loss,
        }


class ClassificationMulticlassModel(ClassificationBase):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        gt_img, gt_label, oh_label = batch['image'], batch['label'].long(), batch['oh_label']
        pr_label = self.model(gt_img.contiguous()).float()
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](pr_label, oh_label) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.metric_values[stage]['pr'].append(pr_label.cpu().detach().argmax(dim=1))
        self.metric_values[stage]['gt'].append(oh_label.cpu().argmax(dim=1))   

        # if stage == 'valid' and self.current_epoch >= 20:
        #     for pr, label, gt in zip(pr_label.cpu().detach(), gt_label, oh_label.cpu()):
        #         print(f'{pr} {label} {gt}')      
        # if self.current_epoch == 8:
        #     self.model.unfreeze_mvit_blocks()

        return {
            'loss': loss,
        }
        
class ClassificationMulticlassDistillationModel(ClassificationBase):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        gt_img, gt_label, oh_label = batch['image'], batch['label'].long(), batch['oh_label']
        pr_label = self.model(gt_img.contiguous())
        # pr_label = [x.float() for x in pr_label]
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](gt_img, pr_label, gt_label) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if hasattr(self.model, 'with_two_heads') and self.model.with_two_heads:
            pr_label = [x.cpu().detach() for x in pr_label]
            pr_label = ((pr_label[0] + pr_label[1]) / 2).argmax(dim=1)
        else:
            pr_label = pr_label.cpu().detach().argmax(dim=1)

        self.metric_values[stage]['pr'].append(pr_label)
        self.metric_values[stage]['gt'].append(oh_label.cpu().argmax(dim=1))   
    
        return {
            'loss': loss,
        }