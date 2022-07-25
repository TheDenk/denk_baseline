import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .utils import instantiate_from_config, get_obj_from_str


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train = instantiate_from_config(config['datasets']['train'])
        if self.train.augs: self.train.augs = get_obj_from_str(self.train.augs)()

        self.valid = instantiate_from_config(config['datasets']['valid'])
        if self.valid.augs: self.valid.augs = get_obj_from_str(self.valid.augs)()

    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.config['common'].get('batch_size', 1),  
                          num_workers=self.config['common'].get('num_workers', 1),
                          drop_last=self.config['common'].get('drop_last', True),
                          pin_memory=self.config['common'].get('pin_memory', True),
                          shuffle=self.config['common'].get('shuffle', True),)

    def val_dataloader(self):
        return DataLoader(self.valid, 
                          batch_size=self.config['common'].get('batch_size', 1),  
                          num_workers=self.config['common'].get('num_workers', 1),
                          drop_last=self.config['common'].get('drop_last', False),
                          pin_memory=self.config['common'].get('pin_memory', True),
                          shuffle=self.config['common'].get('shuffle', False),)