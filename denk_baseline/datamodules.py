import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .utils import instantiate_from_config, get_obj_from_str


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._legacy_loader_params = None

        if 'datasets' in config:
            self.train = instantiate_from_config(config['datasets']['train'])
            if hasattr(self.train, 'augs') and isinstance(self.train.augs, str):
                self.train.augs = get_obj_from_str(self.train.augs)()

            self.valid = config['datasets'].get('valid', False)
            if self.valid:
                self.valid = instantiate_from_config(config['datasets']['valid'])
                if hasattr(self.valid, 'augs') and isinstance(self.valid.augs, str):
                    self.valid.augs = get_obj_from_str(self.valid.augs)()

            self.test = config['datasets'].get('test', False)
            if self.test:
                self.test = instantiate_from_config(config['datasets']['test'])
                if hasattr(self.test, 'augs') and isinstance(self.test.augs, str):
                    self.test.augs = get_obj_from_str(self.test.augs)()
        else:
            datamodule_cfg = config.get('datamodule', {})
            datamodule_params = datamodule_cfg.get('params', {})
            dataset_cfg = datamodule_params.get('dataset')
            if not dataset_cfg:
                raise KeyError("Config must define 'datasets' or 'datamodule.params.dataset'.")
            self.train = instantiate_from_config(dataset_cfg)
            if hasattr(self.train, 'augs') and isinstance(self.train.augs, str):
                self.train.augs = get_obj_from_str(self.train.augs)()
            self.valid = False
            self.test = False
            self._legacy_loader_params = {
                'batch_size': datamodule_params.get('batch_size', 1),
                'num_workers': datamodule_params.get('num_workers', 0),
                'shuffle': True,
                'drop_last': datamodule_params.get('drop_last', False),
                'pin_memory': datamodule_params.get('pin_memory', True),
            }

    def train_dataloader(self):
        if 'dataloaders' in self.config:
            loader_params = self.config['dataloaders']['train']['params']
            sampler_class = get_obj_from_str(self.config['dataloaders']['train']['sampler']['target'])
            sampler_params = self.config['dataloaders']['train']['sampler'].get('params', {})
            return DataLoader(
                self.train,
                sampler=sampler_class(self.train, **sampler_params),
                **loader_params
            )
        return DataLoader(self.train, sampler=RandomSampler(self.train), **self._legacy_loader_params)

    def val_dataloader(self):
        if not self.valid:
            return None
        loader_params = self.config['dataloaders']['valid']['params']
        sampler_class = get_obj_from_str(self.config['dataloaders']['valid']['sampler']['target'])
        sampler_params = self.config['dataloaders']['valid']['sampler'].get('params', {})
        return DataLoader(self.valid, 
                          sampler=sampler_class(self.valid, **sampler_params),
                          **loader_params)

    def test_dataloader(self):
        if not self.test:
            return None
        loader_params = self.config['dataloaders']['test']['params']
        sampler_class = get_obj_from_str(self.config['dataloaders']['test']['sampler']['target'])
        sampler_params = self.config['dataloaders']['test']['sampler'].get('params', {})
        return DataLoader(self.test, 
                          sampler=sampler_class(self.test, **sampler_params),
                          **loader_params)
