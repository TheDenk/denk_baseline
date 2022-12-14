import sys
sys.path.append('./repositories/pytorch-image-models')
sys.path.append('./repositories/segmentation_models.pytorch')

import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from denk_baseline.datamodules import DataModule
from denk_baseline.utils import instantiate_from_config, get_obj_from_str


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./configs/base_config.yaml')
    args = parser.parse_args()
    return args

def save_config(config):
    os.makedirs(config['common']['save_path'], exist_ok=True)
    with open(f"{config['common']['save_path']}/config.yaml", 'w') as file:
        OmegaConf.save(config=config, f=file)

def preprocess_config(config):
    # Right exp dir
    exp_name = config['common'].get('exp_name', 'exp0')
    project_name = config['common'].get('project_name', 'proj0')
    save_dir = config['common'].get('save_dir', 'output')
    config['common']['save_path'] = os.path.join(save_dir, project_name, exp_name)

    # Common params
    # MAX EPOCH
    max_epoch = config['common'].get('max_epoch', False)
    if max_epoch:
        config['trainer']['params']['max_epoch'] = max_epoch
        
        for opt_index, _ in enumerate(config['optimizers']):
            sch = config['optimizers'][opt_index].get('scheduler', False)
            if  sch and 'LinearWarmupCosineAnnealingLR' in sch['target']:
                config['optimizers'][opt_index]['scheduler']['params']['max_epoch'] = max_epoch

    # IMG_SIZE
    img_size = config['common'].get('img_size', False)
    if img_size:
        for stage in ['train', 'valid']:
            for side in ['img_h', 'img_w']:
                config['datasets'][stage]['params'][side] = img_size

    # BATCH_SIZE
    batch_size = config['common'].get('batch_size', False)
    if img_size:
        config['dataloaders']['train']['params']['batch_size'] = batch_size
        config['dataloaders']['valid']['params']['batch_size'] = batch_size

    # NUM_WORKERS
    num_workers = config['common'].get('num_workers', False)
    if img_size:
        config['dataloaders']['train']['params']['num_workers'] = num_workers
        config['dataloaders']['valid']['params']['num_workers'] = num_workers
    
    return config

def parse_loggers(config):
    str_loggers = config.get('loggers', [])
    loggers = []
    for str_logger in str_loggers:
        params = str_logger.get('params', {})
        if 'TensorBoardLogger' in str_logger['target']:
            str_logger['params'] = {
                **params,
                'name': config['common'].get('project_name', 'proj0'),
                'version': config['common'].get('exp_name', 'exp0'),
                'save_dir': 'output',
            }
        elif 'WandbLogger' in str_logger['target']:
            str_logger['params'] = {
                **params,
                'name': config['common'].get('exp_name', 'exp0'),
                'project': config['common'].get('project_name', 'proj0'),
                'save_dir': 'output',
            }

        logger = instantiate_from_config(str_logger)
        loggers.append(logger)
    return loggers if len(loggers) else None


if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config)
    config = preprocess_config(config)
    save_config(config)
    print(OmegaConf.to_yaml(config))
    
    seed_everything(config['common']['seed'], workers=True)

    datamodule = DataModule(config)
    model = get_obj_from_str(config['lightning_model'])(config)

    logger = parse_loggers(config)
    trainer = get_obj_from_str(config['trainer']['target'])(logger=logger, **config['trainer']['params'])

    trainer.fit(model, datamodule) 