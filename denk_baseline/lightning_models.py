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

        ignore_layers = set(config.get('ignore_layers', []))
        if ignore_layers:
            state_dict = {}
            for name, params in ckpt.items():
                if name in ignore_layers:
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
        inputs = batch.get('inputs', batch.get('image'))
        targets = batch.get('targets', batch.get('sg_mask'))
        targets_aux = batch.get('targets_aux', batch.get('oh_mask'))
        preds = self.model(inputs.contiguous())
        # print(gt_img.shape, pr_mask.shape, oh_mask.shape, sg_mask.shape)
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](preds, targets) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_step=False, on_epoch=True, prog_bar=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            index = 0 if self.use_bg[m_name] else 1
            metric_value = self.metrics[m_name](preds[:, index:, :, :], targets_aux[:, index:, :, :])
            self.log(metric_info, metric_value, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'loss': loss,
        }


class SegmentationBinaryModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        inputs = batch.get('inputs', batch.get('image'))
        targets = batch.get('targets', batch.get('mask')).float()
        preds = self.model(inputs.contiguous()).float()
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](preds, targets) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)

        for m_name in self.metrics.keys():
            metric_info = f"{m_name}_{stage}"
            metric_value = self.metrics[m_name](preds, targets)
            self.log(metric_info, metric_value, on_step=False, on_epoch=True, prog_bar=True)              
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
            self.log(f'{m_name}_{stage}', metric_value, on_step=False, on_epoch=True, prog_bar=True)

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
                if mon_value > best_metric:
                    best_metric = mon_value
                    best_thres = thres

            for metric_name, metric_info in metrics_thresholds['metrics'].items():
                metric = get_obj_from_str(metric_info['target'])(**metric_info.get('params', {}), threshold=best_thres)
                metric_value = metric(torch.cat(pr), torch.cat(gt))
                self.log(f'{metric_name}_{stage}', metric_value, on_step=False, on_epoch=True, prog_bar=True)

            self.log(f'{stage}_best_threshold', best_thres, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.calculate_metrics('valid')
    
    def on_train_epoch_end(self):
        self.calculate_metrics('train')

    def on_test_epoch_end(self):
        self.calculate_metrics('test')


class ClassificationBinaryModel(ClassificationBase):
    def _common_step(self, batch, batch_idx, stage):
        with torch.autograd.set_detect_anomaly(True):
            inputs = batch.get('inputs', batch.get('image'))
            targets = batch.get('targets', batch.get('label')).float().unsqueeze(1)
            preds = self.model(inputs.contiguous()).float()

            loss = 0
            for c_name in self.criterions.keys():
                c_loss = self.criterions[c_name](preds, targets) * self.crit_weights[c_name]
                self.log(f'{c_name}_loss_{stage}', c_loss, on_epoch=True, prog_bar=True)
                loss += c_loss
            self.log(f'total_loss_{stage}', loss, on_step=False, on_epoch=True, prog_bar=True)
            
            self.metric_values[stage]['pr'].append(preds.cpu().detach().squeeze())
            self.metric_values[stage]['gt'].append(targets.cpu().detach().squeeze().long())

        return {
            'loss': loss,
        }


class ClassificationMulticlassModel(ClassificationBase):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        inputs = batch.get('inputs', batch.get('image'))
        targets = batch.get('targets', batch.get('label')).long()
        targets_aux = batch.get('targets_aux', batch.get('oh_label'))
        preds = self.model(inputs.contiguous()).float()
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](preds, targets_aux) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True)
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.metric_values[stage]['pr'].append(preds.cpu().detach().argmax(dim=1))
        self.metric_values[stage]['gt'].append(targets_aux.cpu().argmax(dim=1))

        return {
            'loss': loss,
        }


class ClassificationMulticlassWithModelLoss(ClassificationBase):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        inputs = batch.get('inputs', batch.get('image'))
        targets = batch.get('targets', batch.get('label')).long()
        targets_aux = batch.get('targets_aux', batch.get('oh_label'))
        loss, preds = self.model(inputs.contiguous(), labels=targets_aux)
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)
        # print(pr_label.shape)
        # print(pr_label.shape, oh_label.shape)
        targets_aux = targets_aux.mean(1)
        self.metric_values[stage]['pr'].append(preds.cpu().detach().argmax(dim=1))
        self.metric_values[stage]['gt'].append(targets_aux.cpu().argmax(dim=1))

        return {
            'loss': loss,
        }
    
class ClassificationMulticlassDistillationModel(ClassificationBase):
    def __init__(self, config):
        super().__init__(config)

    def _common_step(self, batch, batch_idx, stage):
        inputs = batch.get('inputs', batch.get('image'))
        targets = batch.get('targets', batch.get('label')).long()
        targets_aux = batch.get('targets_aux', batch.get('oh_label'))
        preds = self.model(inputs.contiguous())
        # pr_label = [x.float() for x in pr_label]
        
        loss = 0
        for c_name in self.criterions.keys():
            c_loss = self.criterions[c_name](inputs, preds, targets) * self.crit_weights[c_name]
            self.log(f"{c_name}_loss_{stage}", c_loss, on_epoch=True, prog_bar=True, )
            loss += c_loss
        self.log(f"total_loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if hasattr(self.model, 'with_two_heads') and self.model.with_two_heads:
            preds = [x.cpu().detach() for x in preds]
            preds = ((preds[0] + preds[1]) / 2).argmax(dim=1)
        else:
            preds = preds.cpu().detach().argmax(dim=1)

        self.metric_values[stage]['pr'].append(preds)
        self.metric_values[stage]['gt'].append(targets_aux.cpu().argmax(dim=1))
    
        return {
            'loss': loss,
        }