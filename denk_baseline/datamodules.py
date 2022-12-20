import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .utils import instantiate_from_config, get_obj_from_str


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train = instantiate_from_config(config['datasets']['train'])
        if self.train.augs: self.train.augs = get_obj_from_str(self.train.augs)()

        self. valid = config['datasets'].get('valid', False)
        if self.valid:
            self.valid = instantiate_from_config(config['datasets']['valid'])
            if self.valid.augs: self.valid.augs = get_obj_from_str(self.valid.augs)()

        self.test = config['datasets'].get('test', False)
        if self.test:
            self.test = instantiate_from_config(config['datasets']['test'])
            if self.test.augs: self.test.augs = get_obj_from_str(self.test.augs)()

    def train_dataloader(self):
        loader_params = self.config['dataloaders']['train']['params']
        sampler_class = get_obj_from_str(self.config['dataloaders']['train']['sampler']['target'])
        sampler_params = self.config['dataloaders']['train']['sampler'].get('params', {})
        return DataLoader(self.train, 
                          sampler=sampler_class(self.train, **sampler_params),
                          **loader_params)

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
