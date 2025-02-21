import os
from argparse import ArgumentParser
import glob

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from denk_baseline.datamodules import DataModule
from denk_baseline.utils import instantiate_from_config, get_obj_from_str


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--test', required=False, default=False, type=bool)
    args = parser.parse_args()
    return args


def save_config(config):
    os.makedirs(config['general']['save_path'], exist_ok=True)
    with open(f"{config['general']['save_path']}/config.yaml", 'w') as file:
        OmegaConf.save(config=config, f=file)


def preprocess_config(config):
    exp_name = config['general'].get('exp_name', 'noname_experiment')
    project_name = config['general'].get('project_name', 'noname_project')
    save_dir = config['general'].get('save_dir', 'noname_project_output')
    config['general']['save_path'] = os.path.join(save_dir, project_name, exp_name)
    return config


def parse_loggers(config):
    str_loggers = config.get('loggers', [])
    loggers = {}
    for str_logger in str_loggers:
        params = str_logger.get('params', {})
        if 'TensorBoardLogger' in str_logger['target']:
            str_logger['params'] = {
                **params,
                'name': config['general'].get('project_name', 'proj0'),
                'version': config['general'].get('exp_name', 'exp0'),
                'save_dir': config['general'].get('save_dir', 'output'),
            }
            loggers['tensorboard'] = instantiate_from_config(str_logger)
        elif 'WandbLogger' in str_logger['target']:
            str_logger['params'] = {
                **params,
                'name': config['general'].get('exp_name', 'exp0'),
                'project': config['general'].get('project_name', 'proj0'),
                'save_dir': config['general'].get('save_dir', 'output'),
            }
            loggers['wandb'] = instantiate_from_config(str_logger)
        elif 'clearml' in str_logger['target']:
            from clearml import Task
            _ = Task.init(**str_logger['params'])

    return loggers if len(loggers) else None


def run_experiment(config):
    config = preprocess_config(config)
    save_config(config)
    print(OmegaConf.to_yaml(config))
    
    seed_everything(config['general']['seed'], workers=True)

    datamodule = DataModule(config)
    model = get_obj_from_str(config['lightning_model'])(config)

    loggers = parse_loggers(config)
    trainer = get_obj_from_str(config['trainer']['target'])(logger=[v for _, v in loggers.items()], **config['trainer']['params'])

    trainer.fit(model, datamodule) 
    # if 'wandb' in loggers: loggers['wandb']._experiment.finish()
    extract_models(model, config) # remove lightning prefix


def extract_models(model, config):
    ckpt_paths = glob.glob(config['general']['save_path'] + '/*.ckpt')
    ckpt_paths = list(sorted(ckpt_paths))
    model = model.cpu()

    for ckpt_path in ckpt_paths:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
        
        state_dict = {}
        m_dict = model.state_dict()
        for name in m_dict:
            if 'batch_augs' in name:
                continue
            state_dict[name[6:]] = m_dict[name]
        
        torch.save({
            'state_dict': state_dict,
        }, ckpt_path)


def do_test(config):
    config = preprocess_config(config)
    seed_everything(config['general']['seed'], workers=True)
    datamodule = DataModule(config)
    model = get_obj_from_str(config['lightning_model'])(config)
    loggers = parse_loggers(config)
    trainer = get_obj_from_str(config['trainer']['target'])(logger=[v for _, v in loggers.items()], **config['trainer']['params'])
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    args = parse_args()
    main_config = OmegaConf.load(args.config)
    run_experiment(main_config)
